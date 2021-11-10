from PIL import Image
from flask import Flask, send_file, Response
from flask_wtf import FlaskForm
from wtforms import FileField
from flask_uploads import configure_uploads, IMAGES, UploadSet
from MainScript import Canny
from Configuration import directory_path, clusters, scale_factor
from RenderScript import Canny_Render
from flask_sitemap import Sitemap

app = Flask(__name__)
ext = Sitemap(app=app)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = '/static/img/'
app.config['SECRET_KEY'] = 'thisisasecret'
app.config['UPLOADS_DEFAULT_DEST'] = '/static/img/'
images = UploadSet('images', IMAGES)
configure_uploads(app, images)


class MyForm(FlaskForm):
    image = FileField('image')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        if filename.endswith(".png"):

            image = Image.open(f'{directory_path}static/img/images/{filename}')

            # If the file is a PNG we need to remove the extension to resave it as a jpg
            filename_png, extension = filename.split(".")
            image.convert('RGB').save(f'{directory_path}static/img/images/{filename_png}.jpg', "JPEG")

            # Open the new jpg
            image = Image.open(f'{directory_path}static/img/images/{filename_png}.jpg')

            # We send the filename to the Class for it to overwrite the original image with the output to save server space.
            # "filename" includes the .png extension though that we don't want. Therefore, we need to add the .jpg extension to our now shortened
            # filename_png
            filename = f'{filename_png}'
            pic = Canny(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.remove_background()
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()
            #pic.clean_up()
            svg_path = filename.split(".")[0]

            return send_file(f'{directory_path}static/img/images/{svg_path}.zip', as_attachment=True)

        else:
            filename_split = filename.split(".")
            extension = filename.split(".")[-1]
            filename = ".".join(filename_split[0:-1])
            if extension == "jpg":
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpg')
            else:
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpeg')
            pic = Canny(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.remove_background()
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()
            #pic.clean_up()

            return send_file(f'{directory_path}static/img/images/{filename}.zip', as_attachment=True)

    return render_template('upload2.html', form=form)


@ext.register_generator
def index():
    # Not needed if you set SITEMAP_INCLUDE_RULES_WITHOUT_PARAMS=True
    yield 'index', {}


@app.route('/return-files/<filename>')
def return_files_tut(filename):
    try:
        return send_file(f'static/img/images/{filename}.pdf', attachment_filename='Download.pdf')

    except Exception as e:
        return str(e)


@app.route('/finalanimation', methods=['GET', 'POST'])
def final_animation():
    form = MyForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        if filename.endswith(".png"):

            image = Image.open(f'{directory_path}static/img/images/{filename}')

            # If the file is a PNG we need to remove the extension to resave it as a jpg
            filename_png, extension = filename.split(".")
            image.convert('RGB').save(f'{directory_path}static/img/images/{filename_png}.jpg', "JPEG")

            # Open the new jpg
            image = Image.open(f'{directory_path}static/img/images/{filename_png}.jpg')

            # We send the filename to the Class for it to overwrite the original image with the output to save server space.
            # "filename" includes the .png extension though that we don't want. Therefore, we need to add the .jpg extension to our now shortened
            # filename_png
            filename = f'{filename_png}'
            pic = Canny_Render(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()
            svg_path = filename.split(".")[0]

            return send_file(f'{directory_path}static/img/images/{svg_path}.zip', as_attachment=True)

        else:
            filename_split = filename.split(".")
            extension = filename.split(".")[-1]
            filename = ".".join(filename_split[0:-1])
            if extension == "jpg":
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpg')
            else:
                image = Image.open(f'{directory_path}static/img/images/{filename}.jpeg')
            pic = Canny_Render(user_image=image, name=filename, clusters=clusters, scalar=scale_factor)
            pic.k_means()
            pic.median()
            pic.auto_canny(sigma=0.33)
            pic.collate_svg()

            return render_template('finalimage.html', user_image=f'static/img/images/{filename}.svg', filename=filename)

    return render_template('upload2.html', form=form)


@app.route('/privacy')
def privacy_policy():
    return render_template('privacy.html')


### Blog ###
from flask import render_template, url_for
from flask_flatpages import FlatPages
from flask_frozen import Freezer

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'

app.config.from_object(__name__)
pages = FlatPages(app)
freezer = Freezer(app)


@app.route('/blog')
def blog():
    return render_template('bloghome.html', pages=pages)


@app.route('/<path:path>.html')
def page(path):
    page = pages.get_or_404(path)
    return render_template('page.html', page=page)


@freezer.register_generator
def pagelist():
    for page in pages:
        yield url_for('page', path=page.path)


### Gallery ###
@app.route('/gallery')
def home():
    from os import listdir
    from os.path import isfile, join
    file_list = [f for f in listdir("static/img/images/") if isfile(join("static/img/images/", f))]

    return render_template('gallery.html', file_names=file_list)



if __name__ == '__main__':
    app.run()
