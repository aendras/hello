import fitz  # PyMuPDF
import os

PDF_DIR = "data"
OUTPUT_DIR = "output_images"
DPI = 200  # increase to 300 for higher quality

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pdf_to_images(pdf_path, output_base):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_base, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix)

        image_path = os.path.join(
            pdf_output_dir,
            f"page_{page_num + 1:05d}.png"
        )

        pix.save(image_path)

        if (page_num + 1) % 50 == 0:
            print(f"{pdf_name}: {page_num + 1} pages done")

    doc.close()
    print(f"Finished {pdf_name}")

def process_all_pdfs():
    for file in os.listdir(PDF_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, file)
            pdf_to_images(pdf_path, OUTPUT_DIR)

if __name__ == "__main__":
    process_all_pdfs()
