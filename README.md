# Image Retrieval (Lost & Found)

![Rapido Image Retrieval](readme_files/header.webp)

Rapido Image Retrieval is a Python-based tool for indexing and searching images using feature extraction and similarity search. It supports converting images to JPG format, indexing them efficiently, and retrieving the most similar images given a query image.

## Features

![Lost & Found](readme_files/Figure_1.png)

- Convert various image formats to JPG for consistency
- Index images to create a searchable feature database
- Search for similar images given a query image
- Displays similarity scores and matched images visually
- Supports GPU acceleration (optional)

## Requirements

- Python 3.10
- [Hatch](https://hatch.pypa.io/latest/)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rapido_image_ret.git
   cd rapido_image_ret/src
   ```
   

2. 
    ```
    pip install hatch
    ```

3. 
    ```
    hatch env create
    hatch shell
    ```

4.
    ```
    pip install -r requirements.txt
    ```

## Usage

The main script `index_and_retrieve.py` supports two tasks: **index** and **search**.

### Step 1: Index Images

Before searching, you need to build an index of your image database. Run the script with `task = "index"` to convert images and create the index:

```
python index_and_retrieve.py
```
