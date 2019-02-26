import os

def to_comparemt_format(path, name):
    with open(os.path.join(name), 'w') as writefile:
        files = sorted(os.listdir(path))
        for f in files:
            print(f)
            summary = ''
            with open(os.path.join(path, f), 'r') as file:
                for line in file:
                    summary += line
            summary = summary.replace('\n', ' ') + '\n'
            writefile.write(summary)

def to_sameline(path, new_path):
    files = sorted(os.listdir(path))
    for f in files:
        print(f)
        with open(os.path.join(new_path, f), 'w') as writefile:
            summary = ''
            with open(os.path.join(path, f), 'r') as file:
                for line in file:
                    summary += line
            summary = summary.replace('\n', ' ')
            writefile.write(summary)
