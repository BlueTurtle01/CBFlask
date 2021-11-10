import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import os
import glob
from zipfile import ZipFile
import requests



# directory_path is need as the path required for my local vs the PythonAnywhere server is different.
directory_path = f'/'  # end with a slash


class Canny:
    def __init__(self, name, user_image, clusters, scalar):
        self.file_name = name
        self.imag = user_image
        self.gray = cv2.cvtColor(np.array(self.imag), cv2.COLOR_BGR2GRAY)
        self.clusters = clusters
        self.height, self.width = self.imag.size
        self.scalar = scalar

    def remove_background(self):
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(f'{directory_path}static/img/images/{self.file_name}.jpg', 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': 'h18pKF1C88SuHfKHfTLzvMLH'},
        )
        if response.status_code == requests.codes.ok:
            with open(f'{directory_path}static/img/images/{self.file_name}noBackround.png', 'wb') as out:
                out.write(response.content)

        im = Image.open(f'{directory_path}static/img/images/{self.file_name}noBackround.png')
        new_image = Image.new("RGBA", im.size, "WHITE")  # Create a white rgba background
        new_image.paste(im, (0, 0), im)
        self.imag = new_image.convert('RGB')


    def calculate_filter_size(self):
        # We calculate the number of unique colours to determine the kernel_size. An image with a lot of unique
        # colours requires more blurring, which is achieved with a bigger kernel_size.
        unique_colors = set()
        for i in range(self.imag.size[0]):
            for j in range(self.imag.size[1]):
                pixel = self.imag.getpixel((i, j))
                unique_colors.add(pixel)

        filter_size = int(str(round(len(unique_colors), -3))[0])

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if filter_size % 2 == 0:
            filter_size = max(9, filter_size + 1)

        else:
            filter_size = max(9, filter_size)

        return filter_size

    def k_means(self):
        # Resize the image in the hopes that kmeans and contours can find the edges easier.
        w, h = self.imag.size
        if w > 1000:
            h = int(h * 1000. / w)
            w = 1000
        imag = self.imag.resize((w, h), Image.NEAREST)

        # Dimension of the original image
        cols, rows = imag.size

        # Flatten the image with the new dimensions.
        imag = np.array(imag).reshape(rows * cols, 3)

        # Implement k-means clustering to form k clusters
        kmeans = MiniBatchKMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)


    def median(self):
        self.compressed_image = cv2.resize(self.compressed_image,
                                           dsize=(self.height * self.scalar, self.width * self.scalar),
                                           interpolation=cv2.INTER_NEAREST)

        filter_size = self.calculate_filter_size()
        self.median = cv2.medianBlur(self.compressed_image, filter_size)

        # Create a guide for what colours go where. We will later zip this with the output
        pil_compressed = Image.fromarray(self.median)
        pil_compressed.save(f'{directory_path}static/img/images/{self.file_name}Guide.pdf', "PDF", resolution=100.0)

    def auto_canny(self, sigma=0.33):
        self.canny = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)
        self.canny = cv2.equalizeHist(self.canny)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.5 - sigma) * v))
        upper = int(min(255, (0.8 - sigma) * v))

        self.edged = cv2.Canny(self.canny, lower, upper)
        #edged = cv2.dilate(self.edged, (15, 15), iterations=1)
        #self.edged = cv2.erode(edged, (15, 15), iterations=1)

    @staticmethod
    def calculate_long_contours(contours):
        long_contours = []
        for contour in contours:
            if contour.shape[0] > 0:
                if cv2.contourArea(contour) > 0:
                    long_contours.append(contour)

        return long_contours

    def create_svg(self, contours):
        # For some reason height and width got swapped. As a dirty hack I will just reassign them and reassess if it becomes a bigger problem.
        width = self.height
        height = self.width

        with open(f'{directory_path}static/img/images/{self.file_name}.svg', "w+") as f:
            f.write(f'<svg width="{width}px" height="{height}px" xmlns="http://www.w3.org/2000/svg">')

            for c in contours:
                f.write('<path d="M')
                for i in range(len(c)):
                    x, y = c[i][0]
                    f.write(f"{x} {y} ")
                f.write('" style="stroke:gainsboro;fill:none"/>')  # CSS colour names (lowercase), RGB, or HEX codes can be used.
            f.write("</svg>")

    def create_pdf(self):
        drawing = svg2rlg(f'{directory_path}static/img/images/{self.file_name}.svg')
        renderPDF.drawToFile(drawing, f'{directory_path}static/img/images/{self.file_name}.pdf')

        # Create a ZipFile Object to store the outline and guide in
        with ZipFile(f'{directory_path}static/img/images/{self.file_name}.zip', 'w') as zipObj2:
            # Add multiple files to the zip
            zipObj2.write(f'{directory_path}static/img/images/{self.file_name}.pdf', f'{self.file_name}.pdf')
            zipObj2.write(f'{directory_path}static/img/images/{self.file_name}Guide.pdf', f'{self.file_name}Guide.pdf')


    def collate_svg(self):
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        #contours = self.calculate_long_contours(contours)

        self.create_svg(contours)
        self.create_pdf()

    def clean_up(self):
        for file in glob.glob(f'{directory_path}static/img/images/{self.file_name}'):
            os.remove(file)
