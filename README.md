# Interior Design AI Assistant - Data Pipeline


### Current Phase: Data Foundation (Phase 1 - In Progress)
Collecting and processing interior design images from multiple sources to build a comprehensive dataset for training our AI models.

## 📁 Project Structure

```
interior-design-ai-pipeline/
├── playbootkv_audit.ipynb      # Main Jupyter notebook with pipeline code
├── .env                         # API keys (not tracked in git)
├── .gitignore                   # Git ignore rules
├── interior_design_data/       # Data storage (not tracked)
│   ├── raw_images/             # Original collected images
│   ├── processed_images/       # Resized and optimized images
│   ├── embeddings/             # CLIP embeddings for similarity search
│   ├── metadata/               # Parquet files with image metadata
│   └── interior_design.duckdb  # Main database
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+**
2. **CUDA-capable GPU** (optional but recommended - 10-20x faster)
3. **At least 10GB free disk space**


```

3. Set up API keys in `.env` file:
```env
ROBOFLOW_API_KEY=your_roboflow_private_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HUGGINGFACE_TOKEN=optional_for_private_datasets
```

### Running the Pipeline

Open `playbootkv_audit.ipynb` in Jupyter/VSCode and run the cells sequentially, or run directly:

```python
from playbootkv_audit import DataConfig, DataPipeline

# Quick test with 10 images
config = DataConfig()
pipeline = DataPipeline(config)
pipeline.run_collection_phase(max_samples_per_dataset=10)
```

## 📊 Data Sources

### Currently Integrated

1. **HuggingFace Datasets** (No API key required)
   - `Voxel51/IndoorSceneRecognition` - 15,620 images, 67 indoor categories
   - `hammer888/interior_style_dataset` - Nordic, Modern, Japanese, Luxury styles
   - `keremberke/indoor-scene-classification` - 15,571 indoor scenes

2. **Roboflow** (API key required)
   - `roboflow-100/furniture-ngpea/1` - 689 furniture images
   - `yoloimage-qko0i/interior-design-jsxxo/1` - 10 furniture classes
   - `class-qq9at/interiordesign/1` - 1,737 images, 15 styles

3. **Kaggle** (Credentials required)
   - `robinreni/house-rooms-image-dataset` - All room types
   - `prashantsingh001/bedroom-interior-dataset` - 1,800 bedroom images
   - `galinakg/interior-design-images-and-metadata` - Pinterest data

## 🔧 Pipeline Components

### 1. **DataConfig**
Central configuration managing paths, datasets, and processing parameters.

### 2. **DataCollector**
- Collects images from HuggingFace, Roboflow, and Kaggle
- Handles different dataset formats automatically
- Saves images with unique IDs

### 3. **ImageProcessor**
- Resizes images to 512x512
- Extracts dominant color palettes (5 colors per image)
- Detects room type (living room/bedroom focus)
- Identifies design style (modern, minimalist, etc.)

### 4. **EmbeddingGenerator**
- Uses CLIP model for image embeddings
- Enables similarity search and style matching
- GPU-optimized batch processing

### 5. **DatabaseManager**
- DuckDB for fast queries
- Parquet files for data lake architecture
- Structured storage for metadata and embeddings

## 📈 Current Status

### Completed ✅
- [x] Basic pipeline structure
- [x] HuggingFace dataset integration
- [x] Roboflow API connection
- [x] Database schema design
- [x] Color palette extraction
- [x] CLIP embedding generation

### In Progress 🚧
- [ ] Data collection (targeting 10,000+ images)
- [ ] Room type classifier training
- [ ] Style detection model
- [ ] Kaggle dataset integration

### Known Issues ⚠️
- Pipeline runs slowly on CPU (88+ minutes for full dataset)
- Recommend using GPU or cloud services (RunPod, Colab)
- Some Roboflow datasets require manual access approval

## 🎯 Next Steps (Phase 2)

1. **Train Custom Models**
   - Room type classifier (living room vs bedroom)
   - Style detection (modern, minimalist, etc.)
   - Furniture detection using YOLOv8

2. **Build Recommendation Engine**
   - Vector similarity search using embeddings
   - Style transfer capabilities
   - User preference learning

3. **Vendor Integration**
   - Product matching system
   - Price comparison engine
   - Affiliate API connections


### API Keys Setup

Contact the project lead for:
- Roboflow API key
- Kaggle credentials
- HuggingFace token (if using private datasets)

## 📊 Database Queries

After running the pipeline, explore the data:

```python
from playbootkv_audit import DatabaseManager, DataConfig

config = DataConfig()
db = DatabaseManager(config)

# Get all modern living rooms
modern_living = db.query_by_style('modern')

# Count images by source
stats = db.conn.execute("""
    SELECT source, COUNT(*) as count 
    FROM images 
    GROUP BY source
""").df()
```

## 🐛 Troubleshooting

### "No images collected"
- Check API keys in `.env` file
- Start with HuggingFace only (no API needed)
- Reduce `max_samples_per_dataset`

## 📝 Requirements

Create a `requirements.txt` file:
```
datasets>=2.14.0
transformers>=4.30.0
pillow>=10.0.0
duckdb>=0.9.0
pyarrow>=14.0.0
fastparquet>=0.8.3
roboflow>=1.1.0
kaggle>=1.5.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
tqdm>=4.66.0
pandas>=2.0.0
numpy>=1.24.0
boto3>=1.28.0
python-dotenv>=1.0.0
```

## 📄 License

This project is currently private. All rights reserved.

---
