## Installation
Clone the repo and run `python -m pip install -r requirements.txt` or `python -m pip install -r dev-requirements.txt` in the src folder of the repo, followed by `python -m pip install -e .`. You can then either edit the files or just execute depending on which mode you chose.

## Usage
- Follow "Cloning a Project with Submodules" [from here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
- Put all data like metadata.parquet, metadata_snow.parquet, reference_maps, bigearthnetv2 etc. in repository_root/data/
- Run python files directly or with Docker like:
```
docker build -t prepare_metadata:0.1.0 -f Dockerfile.prepare_metadata ../
docker image prune
docker run --name prepare_metadata_container -v ../data/:/app/data/ joshuacar/prepare_metadata:0.1.0
```
