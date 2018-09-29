# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

import configuration
import inference_wrapper
import caption_generator
import vocabulary
import glob
import re
import io
import json

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#from im2txt import configuration
#from im2txt import inference_wrapper
#from im2txt.inference_utils import caption_generator
#from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "./model",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./output/word_counts.txt", "Text file containing the vocabulary.")

test_filenames = glob.glob('/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/wen/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/*.jpg')
test_images = []
for filename in test_filenames:
    test_images.append(filename)
total_images = len(test_images)
print("Total images = %d" %total_images)
test_images = ",".join(test_images)

tf.flags.DEFINE_string("input_files",test_images,
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")    
#tf.flags.DEFINE_string("input_files", "/home/wen/im2txt_aic_qian/test_image/test1.jpg",
#                       "File pattern or comma-separated list of file patterns "
#                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

##################
    results = []
    results0 = []
    results1 = []
    results2 = []
    count = 1
    for filename in filenames:        
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      image_name_full = os.path.basename(filename)
      print("Captions for %d/30000 image %s:" %(count, image_name_full))
      count = count+1
      b = re.compile(r'.jpg')
      image_name = b.sub('',image_name_full)
      
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = "".join(sentence)
        if i == 0:
            results0.append({
                  "image_id" : image_name,
				  "caption" : sentence,
                  })
    
        if i == 1:
            results1.append({
                  "image_id" : image_name,
				  "caption" : sentence,
                  })
        if i == 2:
            results2.append({
                  "image_id" : image_name,
				  "caption" : sentence,
                  })
        results.append({
          "image_id" : image_name,
		  "caption" : sentence,
          })
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
    
    print ("the length of results is:",len(results))
    print ("the length of results is:",len(results0))
    print ("the length of results is:",len(results1))
    print ("the length of results is:",len(results2))
    


      
#          
    outfile = "/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/wen/im2txt-aic/eval/data/val_results.json"
    outfile0 = "/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/wen/im2txt-aic/eval/data/val_results0.json"
    outfile1 = "/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/wen/im2txt-aic/eval/data/val_results1.json"
    outfile2 = "/media/han/6f586f18-792a-40fd-ada6-59702fb5dabc/wen/im2txt-aic/eval/data/val_results2.json"
          

    with io.open(outfile, 'w', encoding='utf-8') as fd:
        fd.write(unicode(json.dumps(results, ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))
    with io.open(outfile0, 'w', encoding='utf-8') as fd0:
        fd0.write(unicode(json.dumps(results0, ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))
    with io.open(outfile1, 'w', encoding='utf-8') as fd1:
        fd1.write(unicode(json.dumps(results1, ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))
    with io.open(outfile2, 'w', encoding='utf-8') as fd2:
        fd2.write(unicode(json.dumps(results2, ensure_ascii=False,sort_keys=True,indent=2,separators=(',', ': '))))



if __name__ == "__main__":
  tf.app.run()
