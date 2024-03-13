import nnef
import argparse
import os
import numpy as np


MAX_RELATIVE_ERROR = 0.01
MAX_KL_DIVERGENCE = 1
MAX_BAD_PIXEL_RATIO = 0.05


def read_tensor(filename):
    with open(filename) as file:
        return nnef.read_tensor(file)


def softmax(x, axis=1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


def KL_divergence(p, q, axis=1):
    return np.sum(p * np.log2(p / np.maximum(q, 1e-12)), axis=axis)


def relative_difference(reference, output):
    diff = np.sqrt(np.mean(np.square(reference - output)))
    range = np.sqrt(np.mean(np.square(reference)))
    return diff / range


def compare_tensors(reference, output):
    if 'float' in str(reference.dtype):
        return relative_difference(reference, output)
    elif reference.dtype == np.bool:
        return 0.0 if np.all(output == reference) else 1.0
    else:
        assert False


def compare_classification(reference, output):
    p = softmax(output)
    q = softmax(reference)
    return np.mean(KL_divergence(p, q))


def compare_segmentation(reference, output):
    area = reference.shape[2] * reference.shape[3]
    return np.min(np.sum(np.argmax(output, axis=1) == np.argmax(reference, axis=1), axis=(1,2)) / area)


def find_reduction_axis(input, index):
    assert len(input.shape) == len(index.shape)
    axis = None
    for i in range(len(input.shape)):
        if index.shape[i] == 1 and input.shape[i] != 1:
            assert axis is None
            axis = i
    return axis


def compare_folders(reference_output_folder, system_output_folder, metric, extension):
    subdirs = [name for name in os.listdir(reference_output_folder) if os.path.isdir(os.path.join(reference_output_folder, name))]

    cases = 0
    failures = 0
    skipped = 0
    for subdir in sorted(subdirs):
        reference_output_path = os.path.join(reference_output_folder, subdir)
        system_output_path = os.path.join(system_output_folder, subdir)

        if os.path.isdir(system_output_path):
            cases += 1
            print("Comparing test case '{}'...".format(subdir))

            input_names = [filename for filename in os.listdir(reference_output_path) if filename.startswith('input')]
            output_names = [filename for filename in os.listdir(reference_output_path) if filename.startswith('output')]
            for idx, output_name in enumerate(output_names):
                reference_output = read_tensor(os.path.join(reference_output_path, output_name))[0]

                try:
                    system_output = read_tensor(os.path.join(system_output_path, output_name))[0]
                except FileNotFoundError:
                    print("COULD NOT FIND outputs for test case '{}'".format(subdir))
                    skipped += 1
                    break

                assert system_output.dtype == reference_output.dtype, "Data type mismatch for output '{}' (expected {}, found {})". \
                    format(output_name, reference_output.dtype, system_output.dtype)
                assert system_output.shape == reference_output.shape, "Shape mismatch for output '{}' (expected {}, found {})". \
                    format(output_name, reference_output.shape, system_output.shape)

                if metric == 'relative-difference':
                    if 'int' in str(reference_output.dtype):
                        input = read_tensor(os.path.join(reference_output_path, input_names[idx]))[0]
                        axis = find_reduction_axis(input, reference_output)
                        assert axis
                        reference_output = np.take_along_axis(input, reference_output, axis=axis)
                        system_output = np.take_along_axis(input, system_output, axis=axis)

                    cmp = compare_tensors(reference_output, system_output)
                    match = cmp < MAX_RELATIVE_ERROR
                    msg = "({0:.2f}% mean relative error)".format(cmp * 100)
                elif metric == 'classification':
                    cmp = compare_classification(reference_output, system_output)
                    match = cmp < MAX_KL_DIVERGENCE
                    msg = "({0:.2f} mean KL-divergence)".format(cmp)
                elif metric == 'segmentation':
                    cmp = compare_segmentation(reference_output, system_output)
                    match = cmp > (1 - MAX_BAD_PIXEL_RATIO)
                    msg = "({}% bad pixel ratio)".format((1 - cmp) * 100)

                print("\tOutput '{}' {} {}".format(output_name, "matches" if match else "DOES NOT MATCH", msg))

                if not match:
                    failures += 1
        elif not extension:
            cases += 1
            failures += 1
            print("COULD NOT FIND outputs for test case '{}'".format(subdir))
        else:
            skipped += 1

    if failures != 0:
        print("Failed {} of {} test cases (skipped {})".format(failures, cases, skipped))
    else:
        print("Passed all {} test cases (skipped {})".format(cases, skipped))


ap = argparse.ArgumentParser()
ap.add_argument('reference_output_folder', type=str, help='path to the reference outputs folder')
ap.add_argument('system_output_folder', type=str, help='path to the system outputs folder')
ap.add_argument('--metric', type=str, help='metric used for comparison; may be on of '
                                           '{ relative-difference, classification, segmentation }',
                default='relative-difference')
ap.add_argument('--extension', action='store_true', help='whether extensions are being compared')

args = ap.parse_args()

compare_folders(args.reference_output_folder, args.system_output_folder, args.metric, args.extension)
