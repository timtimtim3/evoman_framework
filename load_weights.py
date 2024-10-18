import argparse
import os
import pickle
import numpy as np


def load_and_save_data(load_dir, load_filename, save_dir=None, save_filename=None):
    # Construct the full paths for loading and saving
    load_path = os.path.join(load_dir, load_filename)

    # Set default save directory and filename if not provided
    if save_dir is None:
        save_dir = load_dir  # Save in the same directory as the load directory
    if save_filename is None:
        # Replace the file extension with '.txt'
        save_filename = os.path.splitext(load_filename)[0] + '.txt'
    save_path = os.path.join(save_dir, save_filename)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load the pickle file
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        print(f'Loaded data from {load_path}')
    else:
        print(f'File not found: {load_path}')
        return

    # Save the data as a text file
    # Check if data is a NumPy array; if not, convert it
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], np.ndarray):
        print("select 0")
        array_data = data[0]
    elif isinstance(data, np.ndarray):
        array_data = data
    elif isinstance(data, list):
        print("concat")
        # If it's a list of arrays, you may need to concatenate them or handle accordingly
        array_data = np.concatenate(data)
    else:
        print('Data is not a NumPy array or a list containing arrays. Cannot save as text.')
        return

    # Save the array to a text file
    try:
        np.savetxt(save_path, array_data, fmt='%.18e')
        print(f'Saved data to {save_path}')
    except Exception as e:
        print(f'Error saving data to text file: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a pickle file and save its contents as a text file.')
    parser.add_argument('--load_dir', type=str, default="tournament_models_tim", help='Directory to load the pickle file from.')
    parser.add_argument('--load_filename', type=str, default="[1, 2, 3, 4, 6, 7, 8].pkl", help='Name of the pickle file to load.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the text file. Defaults to --load_dir.')
    parser.add_argument('--save_filename', type=str, default=None,
                        help='Name of the text file to save. Defaults to --load_filename with .txt extension.')

    args = parser.parse_args()

    # Set defaults after parsing
    if args.save_dir is None:
        args.save_dir = args.load_dir
    if args.save_filename is None:
        args.save_filename = os.path.splitext(args.load_filename)[0] + '.txt'

    load_and_save_data(
        load_dir=args.load_dir,
        load_filename=args.load_filename,
        save_dir=args.save_dir,
        save_filename=args.save_filename
    )
