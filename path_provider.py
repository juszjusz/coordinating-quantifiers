import os

from pathlib import Path


class PathProvider:
    def __init__(self, root_path):
        # assert isinstance(root_path, Path)
        # assert root_path.is_absolute() # force client to provide absolute path (for safety reasons)

        self.root_path = root_path
        self.cats_path = self.root_path.joinpath('cats')
        self.lang_path = self.root_path.joinpath('langs')
        self.lang2_path = self.root_path.joinpath('langs2')
        self.matrices_path = self.root_path.joinpath('matrices')
        self.data_path = self.root_path.joinpath('data')

    def get_data_paths(self):
        return list(self.root_path.joinpath('data').glob('step[0-9]*.p'))

    def create_directories(self):
        def create_dir_if_not_exists(path):
            path_as_str = str(path)
            if not os.path.exists(path_as_str):
                os.mkdir(path_as_str)

        create_dir_if_not_exists(self.cats_path)
        create_dir_if_not_exists(self.lang_path)
        create_dir_if_not_exists(self.lang2_path)
        create_dir_if_not_exists(self.matrices_path)
        create_dir_if_not_exists(self.data_path)

    def get_simulation_step_path(self, step):
        return self.data_path.joinpath('step{}.p'.format(step))

    def get_simulation_params_path(self):
        return self.root_path.joinpath('params.p')

    def get_inmem_calc_path(self):
        return self.root_path.joinpath('inmem_calc.p')

    def get_stimuluses_path(self):
        return self.root_path.joinpath('stimuluses.p')

    def create_directory_structure(self):
        os.makedirs(str(self.data_path))


    @staticmethod
    def new_path_provider(path):
        if isinstance(path, str):
            return PathProvider(Path(os.path.abspath(path)))
        if isinstance(path, Path):
            return PathProvider(Path(os.path.abspath(str(path))))