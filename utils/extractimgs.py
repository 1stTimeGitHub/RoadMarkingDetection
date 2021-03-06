import argparse
import logging
import os
import cv2
from tokenize import String

def extract(indir: str, outdir: str, ext: str, n: int):
    """
    Extracts frames from a video. The frames are extracted every n frames.

    Args:
        indir (str): Path to the directory where the videos are located.
        outdir (str): Path of the location where the extracted frames are going to be saved.
        ext (str): Desired extention for the saved frames. Can be either '.jpeg' or '.png'
        n (int): Frequency (in frames) at which the frames are saved.
    """

    assert os.path.isdir(indir) is True, 'Input directory does not exist - ' + indir
    assert ext == '.jpeg' or ext == '.png', 'Extension not supported. It must be .jpeg or .png'

    vid_list = os.listdir(indir)

    #Iterates over the videos in the directory
    for vid in vid_list:
        cap = cv2.VideoCapture(indir + '/' + vid)
        frame_count = 0

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret is False: break

            if frame_count % n == 0:
                imgname = outdir + '/' + vid + '-frame' + str(frame_count) + ext
                logging.info('Writing: ' + imgname)
                cv2.imwrite(imgname, frame)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--video-directory', help='Directory where the videos are stored', dest='dir', required=True)
    parser.add_argument('-o', '--output-images-directory', help='Directory where the iamges extracted from the videos are going to be stored', dest='outdir', required=True)
    parser.add_argument('-e', '--images-extension', help='Extension of the output images. Either .jpeg or .png.', dest='ext', required=True)
    parser.add_argument('-n', '--extract-frame-every-nth-frame', help='Extract a frame every nth frame', dest='n', type=int, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    extract(args.dir, args.outdir, args.ext, args.n)