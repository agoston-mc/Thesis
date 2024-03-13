import sys
import os


operator_domains = {
    'log': (0.0, 1.0),
    'log2': (0.0, 1.0),
    'sqrt': (0.0, 1.0),
    'rsqrt': (0.0, 1.0),
    'pow': (0.0, 1.0),
    'batch_norm': (0.0, 1.0),
    'acosh': (1.0, 2.0),
}


def domain(case):
    for key, value in operator_domains.items():
        if case.startswith(key):
            return value

    return -1.0, 1.0


def list_files(path, ext, recursive=True):
    if recursive:
        return [os.path.join(r, file) for r, d, f in os.walk(path) for file in f if file.endswith(ext)]
    else:
        return [file for file in os.listdir(path) if file.endswith(ext)]


if len(sys.argv) < 3:
    print("source folder and output folder must be supplied")
    exit(-1)

source_path = sys.argv[1]
output_path = sys.argv[2]

filenames = sorted(list_files(source_path, '.nnef')) if os.path.isdir(source_path) else [source_path]

for filename in filenames:
    testcase = os.path.split(filename)[1][:-5]

    print("Generating test case '{}'...".format(testcase))

    low, high = domain(testcase)
    random = "{{'scalar': uniform({low}, {high}), 'integer': integers(0, 64), 'logical': bernoulli(0.5)}}"\
        .format(low=low, high=high)

    print(random)

    testcase = os.path.join(output_path, testcase)

    if not os.path.exists(testcase):
        os.mkdir(testcase)

    os.system("cp {} {}/graph.nnef".format(filename, testcase))
    os.system("python3 -m nnef_tools.generate {case} --inputs --weights --random \"{rnd}\"".format(rnd=random, case=testcase))
    os.system("python3 -m nnef_tools.execute {case} --format nnef --input-path {case} --output-path {case} --statistics {case}/graph.stats".format(case=testcase))
