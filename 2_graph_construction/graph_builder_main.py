import os
from joblib import Parallel, delayed
from populateGraphWithCallStack import createWebGraphWithCallStack
lst = [0]
def process_folder(folder):
    fold = "server/output/" + folder
    graph_pdf_path = os.path.join(fold, "graph.pdf")

    # Check if the graph PDF file already exists
    if os.path.exists(graph_pdf_path):
        lst[0] += 1
        with open("graph_logs.txt", "w") as count_file:
            count_file.write(str(lst[0]))
        return None  # Skip this folder

    print("Starting graph-genration: website:", folder)
    try:
        createWebGraphWithCallStack(folder)
        print("Completed graph-genration: website:", folder)
        # Increment the counter after successful processing
        lst[0] += 1
        # Log the updated count to a file
        with open("logs/graph_logs.txt", "w") as count_file:
            count_file.write(str(lst[0]))
    except Exception as e:
        print("Crashed graph-genration: website:", folder, e)
    return folder

def main():
    folders = os.listdir("server/output")
    # folders.remove(".DS_Store")
    num_jobs = len(folders)
    num_parallel_jobs = 1  # Use all available CPU cores

    results = Parallel(n_jobs=num_parallel_jobs)(
        delayed(process_folder)(folder) for folder in folders
    )

if __name__ == "__main__":
    main()
