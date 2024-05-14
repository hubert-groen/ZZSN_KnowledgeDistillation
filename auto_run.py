import subprocess

def run_training(index):
    command = f"python main.py --train_index {index}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

def main():
    start_index = 25
    end_index = 50

    for index in range(start_index, end_index + 1):
        run_training(index)

if __name__ == "__main__":
    main()
