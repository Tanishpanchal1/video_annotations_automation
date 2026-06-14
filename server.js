const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
// Serve the public directory
app.use(express.static(path.join(__dirname, 'public')));
// Serve the main directory for video file playback/assets
app.use('/workspace', express.static(__dirname));

// Multer storage configuration for uploads
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// API: List assets in the workspace
app.get('/api/assets', (req, res) => {
  try {
    const files = fs.readdirSync(__dirname);
    const images = [];
    const audios = [];
    const videos = [];

    // Add workspace root files
    files.forEach(file => {
      const ext = path.extname(file).toLowerCase();
      const fullPath = path.join(__dirname, file);
      if (fs.statSync(fullPath).isDirectory()) return;

      if (['.png', '.jpg', '.jpeg', '.webp'].includes(ext)) {
        images.push({ name: file, type: 'workspace', path: file });
      } else if (['.mp3', '.wav', '.mpeg', '.m4a'].includes(ext)) {
        audios.push({ name: file, type: 'workspace', path: file });
      } else if (['.mp4', '.mov'].includes(ext)) {
        videos.push({ name: file, type: 'workspace', path: file });
      }
    });

    // Add uploaded files
    if (fs.existsSync(uploadDir)) {
      const uploadedFiles = fs.readdirSync(uploadDir);
      uploadedFiles.forEach(file => {
        const ext = path.extname(file).toLowerCase();
        const relativePath = path.join('uploads', file).replace(/\\/g, '/');
        if (['.png', '.jpg', '.jpeg', '.webp'].includes(ext)) {
          images.push({ name: file, type: 'upload', path: relativePath });
        } else if (['.mp3', '.wav', '.mpeg', '.m4a'].includes(ext)) {
          audios.push({ name: file, type: 'upload', path: relativePath });
        }
      });
    }

    res.json({ images, audios, videos });
  } catch (err) {
    res.status(500).json({ error: 'Failed to read directory', details: err.message });
  }
});

// API: Handle file uploads
app.post('/api/upload', upload.fields([
  { name: 'image', maxCount: 1 },
  { name: 'audio', maxCount: 1 }
]), (req, res) => {
  const response = {};
  if (req.files['image']) {
    response.image = path.join('uploads', req.files['image'][0].filename).replace(/\\/g, '/');
  }
  if (req.files['audio']) {
    response.audio = path.join('uploads', req.files['audio'][0].filename).replace(/\\/g, '/');
  }
  res.json(response);
});

// SSE API: Stream Python execution progress
app.get('/api/stream-progress', (req, res) => {
  const { image, audio } = req.query;

  if (!image || !audio) {
    return res.status(400).json({ error: 'Parameters image and audio are required' });
  }

  // Set up SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  // Resolve absolute paths
  const absoluteImage = path.isAbsolute(image) ? image : path.resolve(__dirname, image);
  const absoluteAudio = path.isAbsolute(audio) ? audio : path.resolve(__dirname, audio);

  // Validate files exist
  if (!fs.existsSync(absoluteImage)) {
    sendSSE(res, { type: 'error', message: `Image not found: ${image}` });
    return res.end();
  }
  if (!fs.existsSync(absoluteAudio)) {
    sendSSE(res, { type: 'error', message: `Audio not found: ${audio}` });
    return res.end();
  }

  // Generate unique output filename
  const timestamp = Date.now();
  const outputFileName = `annotated_output_${timestamp}.mp4`;
  const absoluteOutput = path.join(__dirname, outputFileName);

  // Python executable paths
  const pythonExec = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
  const scriptPath = path.join(__dirname, 'annotation_pipeline.py');

  sendSSE(res, { type: 'log', message: `Spawning Python script with target output: ${outputFileName}...` });

  const args = [scriptPath, '--image', absoluteImage, '--audio', absoluteAudio, '--output', absoluteOutput];
  const pyProcess = spawn(pythonExec, args, { cwd: __dirname });

  let buffer = '';

  pyProcess.stdout.on('data', (data) => {
    buffer += data.toString();
    let lines = buffer.split('\n');
    buffer = lines.pop(); // Keep last incomplete line

    lines.forEach(line => {
      const cleanLine = line.trim();
      if (!cleanLine) return;

      if (cleanLine.startsWith('PROGRESS:')) {
        try {
          const progressData = JSON.parse(cleanLine.substring(9));
          sendSSE(res, { type: 'progress', ...progressData });
        } catch (e) {
          sendSSE(res, { type: 'log', message: `Failed to parse progress JSON: ${cleanLine}` });
        }
      } else {
        sendSSE(res, { type: 'log', message: cleanLine });
      }
    });
  });

  pyProcess.stderr.on('data', (data) => {
    const lines = data.toString().split('\n');
    lines.forEach(line => {
      const cleanLine = line.trim();
      if (cleanLine) {
        sendSSE(res, { type: 'log', message: `[stderr] ${cleanLine}` });
      }
    });
  });

  pyProcess.on('close', (code) => {
    if (code === 0) {
      sendSSE(res, { type: 'complete', output: outputFileName });
    } else {
      sendSSE(res, { type: 'error', message: `Pipeline exited with code ${code}` });
    }
    res.end();
  });

  // Handle client disconnect
  req.on('close', () => {
    if (pyProcess) {
      pyProcess.kill();
    }
  });
});

function sendSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
