import os


def process_raw_data(
    input_dir="data/raw", output_dir="data/processed"
):
  """Reads raw text and markdown files from the raw folder, cleans them,

  and saves the optimized versions into the processed folder for the RAG engine.
  """
  # Ensure output directory exists
  os.makedirs(output_dir, exist_ok=True)

  if not os.path.exists(input_dir):
    print(f"Directory '{input_dir}' does not exist.")
    return

  # List all files in the raw directory
  raw_files = os.listdir(input_dir)

  if not raw_files:
    print(
        f"Warning: The '{input_dir}' folder is empty. Please place your raw"
        " files (like context.txt) there."
    )
    return

  processed_count = 0

  for filename in raw_files:
    # Process only text or markdown files
    if filename.endswith(".txt") or filename.endswith(".md"):
      input_path = os.path.join(input_dir, filename)
      output_filename = f"clean_{filename}"
      output_path = os.path.join(output_dir, output_filename)

      # Read raw content with UTF-8 encoding
      with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

      # Data cleaning: remove trailing spaces, normalize blank lines
      cleaned_lines = [line.strip() for line in content.splitlines()]
      cleaned_content = "\n".join(
          [line for line in cleaned_lines if line]
      )  # Removes empty lines

      # Save processed content
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

      print(
          f"Successfully processed: '{filename}' -> Saved to"
          f" 'data/processed/{output_filename}'"
      )
      processed_count += 1

  if processed_count == 0:
    print("No valid .txt or .md files found in the 'data/raw/' directory.")
  else:
    print(
        f"\nData processing complete! {processed_count} file(s) ready for"
        " rag_engine.py."
    )


if __name__ == "__main__":
  print("=== Running Data Processor Pipeline ===")
  process_raw_data()