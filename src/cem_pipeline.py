# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:10:20 2025

@author: Lenovo
"""


# import os, sys, subprocess

# scripts = [
#     r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files\GoogleplayScrapper.py",
#     r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\Final Codes\Keyword_Extraction_Code.py",
#     r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\Final Codes\Sentimet_Analysis_Code.py",
#     r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\Final Codes\Topic_Modeling_Code.py",
# ]

# for sp in scripts:
#     cwd = os.path.dirname(sp)
#     print(f"\nRunning {os.path.basename(sp)}")
#     subprocess.run([sys.executable, sp], check=True, cwd=cwd)

# print("\nAll scripts finished.")


import os, subprocess, time

scraper_dir = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\scrapper_files"
final_dir   = r"D:\CATALYTICS\ECommerce-Watch-main\ECommerce-Watch-main\Final Codes"

scripts = [
    ("GoogleplayScrapper.py", scraper_dir),
    ("Keyword_Extraction_Code.py", final_dir),
    ("Sentimet_Analysis_Code.py", final_dir),
    ("Topic_Modeling_Code.py", final_dir)
]

print("\nStarting CEM pipeline\n")
start = time.time()

for file, folder in scripts:
    print(f"Running {file}")
    try:
        subprocess.run(["python", file], check=True, cwd=folder)
        print(f"✅ Completed {file}\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {file}")
        print(f"Error: {e}\n")

print(f"Pipeline finished in {time.time() - start:.1f}s")
