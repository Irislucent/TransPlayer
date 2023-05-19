This is the codebase of the paper titled **TransPlayer: Timbre Style Transfer with Flexible Timbre Control**.

[Click me](https://irislucent.github.io/TransPlayer-demos/) to listen to the demos.

- ./dataset: To train with your own data, you first need to preprocess the data using the scripts under ./dataset.
    - ./dataset/crop_data.py: Crop your .wav files into chunks. By default, every chunk is 200 seconds long.
    - ./dataset/preprocess_cqt.py: Extract 84 CQT features from the audio chunks.
    - ./dataset/stereo2mono.py: Only in case your wave files are stereo, you'll want them to be mono at training.


- ./autoencoder: Directory containing the training code.
    - How to train: 
    ```
    cd autoencoder
    python -m train [--data_dir xxxx/directory_of_your_wav_and_npy_files] [--save_dir yyyy/directory_to_save_your_model]
    ```
    - How to perform inference:
    ```
    cd autoencoder
    python -m inference [--feature_path xxxx/path_to_the_original_cqt.npy] [--org "instrument_name_A"] [--trg "instrument_name_B"] [--cp_path yyyy/path_to_your_model]
    ```

- How to generate the waveform
  - [Diffwave implementation](https://github.com/lmnt-com/diffwave) (can't redistribute it here for license reasons)
  - To train Diffwave on CQT, you probably want to tweak the code a bit, particularly in the way the dataset is loaded (we don't have to call torchaudio in every iteration, it doesn't help with constant-Q transform anyways), and the input data parameters (the implementation is meant for Mel-spectrograms, please make sure the parameters match with your CQT)
