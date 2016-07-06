from argparse import ArgumentParser

import concurrent.futures
import urllib.request
import os


def collect_images_from_urls(url_filepath, target_folder, image_class_name):
    """Gather all the images collecting available urls from input file

    Parameters
    ----------
    url_filepath : str
        Path to the file containing URLs of images

    target_folder : str
        Path to the folder where downloaded images will
        be saved.

    image_class_name : str
        The prefix of all saved image files. Most likely, this
        name prefix should correspond to the name of the
        class the images belong to.

    Return
    ------
    int
        Return the number of images collected from urls
    """

    def get_img_from_url(index, url):
        """Closure function invoked by each running downloading Thread"""
        try:
            with urllib.request.urlopen(url) as response:
                if response.headers.get_content_maintype() == 'image':
                    image_filename = image_filename_prefix.format(name=image_class_name,
                                                                  counter=index,
                                                                  ext=response.headers.get_content_subtype())
                    image_filepath = os.path.join(target_folder, image_filename)
                    with open(image_filepath, 'wb') as image_file:
                        image_file.write(response.read())

                print('Fetched URL {}'.format(index))

        except urllib.request.HTTPError:
            pass
        except Exception:
            pass

    image_filename_prefix = '{name}_{counter}.{ext}'
    list_of_urls = list()
    with open(url_filepath) as url_file:
        for url in url_file:
            url = url.strip()
            list_of_urls.append(url)

    print('Collected {} total URLS'.format(len(list_of_urls)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as thread_pool:
        for idx, url in enumerate(list_of_urls):
            thread_pool.submit(get_img_from_url, idx, url)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--url_file', dest='url_file', type=str,
                        help='Path to the input URL file')

    # parser.add_argument('--save_valid', dest='save_valid_urls',
    #                     type=bool, default=True,
    #                     help='Flag to decide if valid urls should be saved into a new file.')

    args = parser.parse_args()

    target_folder_path, _ = os.path.split(os.path.abspath(args.url_file))
    _, class_name = os.path.split(target_folder_path)

    collect_images_from_urls(url_filepath=args.url_file,
                             target_folder=target_folder_path,
                             image_class_name=class_name)




