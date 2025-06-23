# 🌱 Weed Detection API

A simple Flask backend API for UAV weed detection using a trained U-Net model.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-cors
```

### 2. Start the API Server
```bash
cd prueba-u-net-2
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open the Test Interface
Open `index.html` in your browser to use the web interface, or use the API endpoints directly.

## 📋 API Endpoints

### Health Check
```http
GET /health
```
Returns API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Upload and Process Image
```http
POST /upload
```
Upload an image for weed detection processing.

**Request:**
- `Content-Type: multipart/form-data`
- `image`: Image file (JPG, PNG, BMP)
- `threshold`: Detection threshold (optional, default: 0.98)

**Response:**
```json
{
  "image_id": "uuid-string",
  "timestamp": "20240115_103000",
  "status": "completed",
  "threshold": 0.98,
  "statistics": {
    "weed_coverage_percent": 12.5,
    "weed_pixels": 125000,
    "total_pixels": 1000000
  },
  "files": {
    "input": "input_filename.jpg",
    "prediction": "prediction_filename.png",
    "overlay_borders": "overlay_filename.png"
  }
}
```

### Get Result Image
```http
GET /result/<image_id>/<result_type>
```

**Parameters:**
- `image_id`: Unique identifier from upload response
- `result_type`: One of:
  - `input` - Original uploaded image
  - `prediction` - 4-panel prediction visualization
  - `overlay_borders` - Original image with green border overlays

**Response:** Image file (PNG/JPG)

### Get Results Info
```http
GET /results/<image_id>
```
Get information about available results for an image.

**Response:**
```json
{
  "image_id": "uuid-string",
  "available_results": ["input", "prediction", "overlay_borders"],
  "total_files": 3
}
```

### Cleanup Old Files
```http
POST /cleanup
```
Remove files older than 1 hour (maintenance endpoint).

## 🧪 Testing with cURL

### Upload an image:
```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  -F "threshold=0.95" \
  http://localhost:5000/upload
```

### Get the result:
```bash
curl "http://localhost:5000/result/{image_id}/overlay_borders" \
  --output result.png
```

## 🎯 Usage Examples

### Python Client Example:
```python
import requests

# Upload image
with open('test_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'threshold': 0.98}
    response = requests.post('http://localhost:5000/upload', 
                           files=files, data=data)
    result = response.json()

# Get result image
image_id = result['image_id']
img_response = requests.get(f'http://localhost:5000/result/{image_id}/overlay_borders')
with open('result.png', 'wb') as f:
    f.write(img_response.content)
```

### JavaScript/Fetch Example:
```javascript
// Upload image
const formData = new FormData();
formData.append('image', imageFile);
formData.append('threshold', '0.98');

const response = await fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log('Weed coverage:', result.statistics.weed_coverage_percent + '%');
```

## 📁 File Structure

The API creates two directories:
- `api_uploads/` - Stores uploaded images
- `api_results/` - Stores processed results

Files are automatically named with timestamps and unique IDs:
- `20240115_103000_uuid_input.jpg`
- `20240115_103000_uuid_prediction.png`
- `20240115_103000_uuid_overlay_borders.png`

## ⚙️ Configuration

Edit the configuration at the top of `app.py`:

```python
MODEL_PATH = 'training_output/checkpoints/best_model.pth'
CONFIG_PATH = 'configV9.0.json'
DEFAULT_THRESHOLD = 0.98
```

## 🛠️ Features

- ✅ **Drag & Drop Interface** - Easy image upload
- ✅ **Real-time Processing** - Live status updates
- ✅ **Multiple Visualizations** - Prediction grids and border overlays
- ✅ **Statistics** - Weed coverage percentages and pixel counts
- ✅ **Configurable Threshold** - Adjust detection sensitivity
- ✅ **Auto Cleanup** - Remove old files automatically
- ✅ **CORS Enabled** - Works with frontend frameworks
- ✅ **Error Handling** - Comprehensive error messages

## 🔧 Troubleshooting

### Model Loading Issues:
- Ensure `training_output/checkpoints/best_model.pth` exists
- Check that `configV9.0.json` is present
- Verify all dependencies are installed

### Memory Issues:
- Large images may require significant RAM
- Consider reducing image size for faster processing
- Monitor system resources during processing

### Network Issues:
- Check that port 5000 is available
- Ensure firewall allows local connections
- For remote access, change `host='0.0.0.0'` in `app.run()`

## 📊 Performance

- **Processing Time**: ~30-60 seconds for 5280×3956 images
- **Memory Usage**: ~6-8GB during processing
- **Patch Processing**: 108 patches per full-resolution image
- **Device Support**: CUDA, MPS (Apple Silicon), CPU

## 🔒 Security Note

This is a development/testing API. For production use, consider:
- Authentication/authorization
- File size limits
- Rate limiting
- Input validation
- Secure file handling 