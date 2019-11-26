#!/usr/bin/env python3

from datetime import datetime

import atexit
import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='1558M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=20,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    input_file='',
    use_input_delimiters=True,
    input_delimiter="|",
    generate_session_file=True,
):
    """
    Interactively run the model
    :model_name=1558M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    generated_content = ""
    content_list = []

    if generate_session_file:
        def save_session():
            session_filename = datetime.today().strftime("%Y%m%d%H%M%S") + "_gpt2_session.txt"
            with open(session_filename, "w") as session_file:
                if use_input_delimiters:
                    delimited_content = input_delimiter.join(content_list)
                    session_file.write(delimited_content)
                else:
                    session_file.write(generated_content)

        atexit.register(save_session)

    def recursive_input(prompt):
        contents = []

        do_prompt = False
        if prompt:
            do_prompt = True

        while True:
            try:
                if do_prompt:
                    line = input(prompt)
                    do_prompt = False
                else:
                    line = input()
            except EOFError:
                print("=" * 40)
                print("Processing input.")
                print("=" * 40)
                break
            contents.append(line)
        
        return "\n".join(contents)

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        starting_text = ""
        if input_file:
            if os.path.exists(input_file):
                with open(input_file, "r") as file:
                    starting_text = file.read()
                starting_text = starting_text.replace(input_delimiter, "")
            else:
                print("Input file could not be read: " + input_file)
        
        text = ""
        iteration_count = 0
        while True:
            if generated_content:
                print("Do you want to undo the previous iteration? y/n (default is no)")
                yes_no = input()
                if yes_no:
                    if "y" in yes_no.lower()[0]:
                        model_content = content_list.pop()
                        user_content = content_list.pop()
                        text = content_list[-1]
                        generated_content = "".join(content_list)
                        print(generated_content)
                        print("=" * 40 + "\n")
                        print("")
                        print("Removed user content: \n" + user_content + "\n")
                        print("Removed model content: " + model_content)
                        print("")
                        print("=" * 40 + "\n")

            if starting_text:
                raw_text = starting_text
                starting_text = ""
            else:
                print("Paste or type your content. You can enter multiple lines. Press Ctrl+D to continue. >>>")
                raw_text = recursive_input(text)

            while not raw_text:
                print('Do you want the model to generate more text? y/n (default is yes)')
                yes_no = input()
                if yes_no:
                    if "n" in yes_no.lower()[0]:
                        print("Paste or type your content. You can enter multiple lines. Press Ctrl+D to continue. >>>")
                        raw_text = recursive_input(text)
                else:
                    break

            total_sample = generated_content + raw_text
            context_tokens = enc.encode(total_sample)
            out = sess.run(output, feed_dict={context: [context_tokens for _ in range(batch_size)]})[:, len(context_tokens):]
            text = enc.decode(out[0])
            content_list.append(raw_text)
            content_list.append(text)
            generated_content += raw_text + text
            iteration_count += 1
            print(("Iteration " + str(iteration_count) + " " + "=" * 40)[:40])
            print("\n" + generated_content + "\n")
            print("=" * 40 + "\n")

        print("All done!")
        exit()

if __name__ == '__main__':
    fire.Fire(interact_model)

