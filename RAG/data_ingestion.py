from docling.datamodel.base_models import InputFormat
from docling.document_converter import ImageFormatOption, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions 
)
from llama_index.core import SimpleDirectoryReader
import os

from llama_index.readers.docling import DoclingReader 
import streamlit as st

from pathlib import Path
import pikepdf

@st.cache_resource
def config_docling():
    #----------Docling Parser--------------
    ocr_options = EasyOcrOptions()
    ocr_options.lang = ["en"]
    ocr_options.use_gpu = True

    ocr_pipeline_config = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        verbose=False
    )

    #-----OCR for Image and PDF--------------
    docling_format_options = {
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=ocr_pipeline_config
        ),
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=ocr_pipeline_config
        )
    }

    docling_reader = DoclingReader(
        format_options=docling_format_options
    )

    file_extractor_map = {
        ".pdf": docling_reader,
        ".docx": docling_reader,
        ".pptx": docling_reader,
        ".xlsx": docling_reader,
        ".html": docling_reader,
        ".csv": docling_reader,
        
        # Image formats to be OCR'd by Docling
        ".png": docling_reader,
        ".jpg": docling_reader,
        ".jpeg": docling_reader,
        ".gif": docling_reader,
    }

    exclude_extensions = ["*.zip"]
    return file_extractor_map, exclude_extensions



def enforce_mediabox_explicit(pdf_path, output_path, default_box=[0,0,595,842]):
    """Ensures every page has an explicit /MediaBox defined (even if inherited)."""
    pdf = pikepdf.open(pdf_path, allow_overwriting_input=True)
    for page in pdf.pages:
        box = None
        try:
            box = page.obj.get("/MediaBox")
            if box is None:
                box = page.page_dict.get("/MediaBox")  # fallback
        except Exception:
            pass

        if box is None:
            page.obj["/MediaBox"] = pikepdf.Array(default_box)
        else:
            page.obj["/MediaBox"] = pikepdf.Array(list(box))

    pdf.save(output_path)
    print(f"✅ Rewritten MediaBoxes in: {output_path}")

def fix_all_pdfs(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".pdf"):
                input_path = os.path.join(dirpath, file)
                output_path = os.path.join(dirpath, f"{os.path.splitext(file)[0]}.pdf")
                try:
                    enforce_mediabox_explicit(input_path, output_path)
                except Exception as e:
                    print(f"❌ Failed on {input_path}: {e}")

import tempfile
import zipfile
def process_pipeline(input_dir):
    file_extractor_map, exclude_extensions = config_docling()
    fix_all_pdfs(input_dir)

    base_input_path = Path(input_dir).resolve()
    all_documents = []

    with tempfile.TemporaryDirectory() as temp_root:
        print(f"Created temporary root: {temp_root}")
        temp_root_path = Path(temp_root).resolve()

        extracted_at_least_one_file = False

        for zip_path in Path(input_dir).rglob("*.zip"):
            absolute_zip_parent = zip_path.parent.resolve()
            relative_parent_dir = absolute_zip_parent.relative_to(base_input_path)
            target_extract_dir = temp_root_path / relative_parent_dir / f"_temp_extract_{zip_path.stem}"
            target_extract_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Extracting: {zip_path} -> {target_extract_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_extract_dir)
            
            extracted_at_least_one_file = True
        
        if extracted_at_least_one_file:
            print("Loading documents from extracted zip contents...")
            zip_content_reader = SimpleDirectoryReader(
                input_dir=str(temp_root_path),
                file_extractor=file_extractor_map,
                recursive=True
            )
            zip_documents = zip_content_reader.load_data(show_progress=True)
            for doc in zip_documents:
                temp_file_path = Path(doc.metadata["file_path"]).resolve()
                relative_doc_path = temp_file_path.relative_to(temp_root_path)
                correct_path = base_input_path / relative_doc_path
                doc.metadata["file_path"] = str(correct_path)
                doc.metadata["file_name"] = correct_path.name
                
            all_documents.extend(zip_documents)

    #-----------Non-zip Documents---------------------
    print(f"Loading non-zip files from main directory: {input_dir}")
    main_reader = SimpleDirectoryReader(
        input_dir=input_dir,
        file_extractor=file_extractor_map,
        exclude=exclude_extensions,  
        recursive=True
    )
    main_documents = main_reader.load_data(show_progress=True)
    all_documents.extend(main_documents)
    return all_documents

