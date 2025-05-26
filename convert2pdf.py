import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
import matplotlib.image as mpimg

png_files = sorted(glob.glob('/Users/shamanthk/Documents/Spring 2025/iomics/focused/supervised_clas_exp_plots/*.png'))
output_pdf = 'combined_task01_clas_plots.pdf'

with PdfPages(output_pdf) as pdf:
    for png_file in png_files:
        img = mpimg.imread(png_file)
        height, width = img.shape[:2]
        # DPI value
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

print(f"Saved {len(png_files)} images into {output_pdf}")
