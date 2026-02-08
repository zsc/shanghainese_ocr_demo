#!/usr/bin/env python3
"""
批量处理词典页面
"""

import os
import sys
from process_dictionary_page import process_dictionary_page


def main():
    # 处理前十页
    for page_num in range(1, 11):
        input_image = f"page_{page_num:02d}.png"
        output_dir = f"extracted/page_{page_num:02d}"
        
        if not os.path.exists(input_image):
            print(f"Skipping {input_image} (not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {input_image}")
        print(f"{'='*60}")
        
        try:
            result_paths = process_dictionary_page(input_image, output_dir)
            print(f"✅ Page {page_num}: {len(result_paths)} boxes extracted")
        except Exception as e:
            print(f"❌ Page {page_num} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Batch processing completed!")


if __name__ == "__main__":
    main()
