import logging as log
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log
import time

from inference import Inference
from input_feeder import InputFeeder

MODEL = "./models/"

def build_argparser():

    parser = ArgumentParser()

    parser.add_argument('-i', '--input', default=0, help="Path to the input video")
    parser.add_argument('-m', '--model', default=MODEL, help="Path to the input video")
    
    
    return parser


def main(args):
    inference = Inference(args.model)
    inference.load_model()
    
    input = args.input
    
    if input == 0:
        input_feeder = InputFeeder('cam', input)
    elif input.endswith('.jpg') or input.endswith('.jpeg') or input.endswith('.bmp'):
        input_feeder = InputFeeder('image', input)
    else:
        input_feeder = InputFeeder('video', input)
    
    frames = 0
        
    for ret, frame in input_feeder.next_batch():
        
        if not ret:
            break
            
        frames += 1
        
        key = cv2.waitKey(60)
        if key == 27:
            break
        
        outputs  = inference.predict(frame)
        
        inference.preprocess_output(outputs)
        

            
    input_feeder.close()
    

if __name__ == '__main__':
    
    log.info('Start...')

    args = build_argparser().parse_args()

    main(args)

    log.info('End...')