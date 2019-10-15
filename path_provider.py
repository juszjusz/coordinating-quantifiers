import os
import re
import time
import zipfile

from pathlib import Path


class PathProvider:
    def __init__(self, root_path):
        assert isinstance(root_path, Path)
        assert root_path.is_absolute() # force client to provide absolute path (for safety reasons)

        self.root_path = root_path
        self.cats_path = self.root_path.joinpath('cats')
        self.lang_path = self.root_path.joinpath('langs')
        self.lang2_path = self.root_path.joinpath('langs2')
        self.matrices_path = self.root_path.joinpath('matrices')
        self.data_path = self.root_path.joinpath('data')

    def get_sorted_paths(self):
        # compare files stepA.p, stepB.p by its numerals A, B
        def cmp(path1, path2):
            path1no = int(re.search('step(\d+)\.p', path1.name).group(1))
            path2no = int(re.search('step(\d+)\.p', path2.name).group(1))
            return path1no - path2no

        data_path = self.root_path.joinpath('data')
        data_paths = list(data_path.glob('step[0-9]*.p'))
        return sorted(data_paths, cmp=cmp)

    # removes all files in simulation_results directory storing data files in zip fie (to avoid unwanted lost of simulation data)
    # def remove_data_files(self):
    #     def remove_files(subdirectory, deleted_file_name_pattern):
    #         for file_to_deletion in subdirectory.glob(deleted_file_name_pattern):
    #             Path.unlink(file_to_deletion)
    #
    #     zipped_data_dir = self.root_path.joinpath('data{}.zip'.format(int(round(time.time() * 1000))))
    #     zipped_data = zipfile.ZipFile(str(zipped_data_dir), 'w', zipfile.ZIP_DEFLATED)
    #     for data_file in self.root_path.joinpath('data').glob('*'):
    #         zipped_data.write(str(data_file))
    #     zipped_data.close()
    #
    #     remove_files(self.cats_path, 'categories[0-9]*_[0-9]*.png')
    #     remove_files(self.lang_path, 'language[0-9]*_[0-9]*.png')
    #     remove_files(self.lang2_path, 'language[0-9]*_[0-9]*.png')
    #     remove_files(self.matrices_path, 'matrix[0-9]*_[0-9]*.png')
    #     remove_files(self.data_path, 'step[0-9]*.p')
    #     remove_files(self.data_path, '[0-9]*.p')

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

    def create_data_path(self, step):
        return self.data_path.joinpath('step{}.p'.format(step))

    def create_directory_structure(self):
        os.mkdir(str(self.run_path))
        os.mkdir(str(self.data_path))

    @staticmethod
    def new_path_provider(path):
        if isinstance(path, str):
            return PathProvider(Path(os.path.abspath(path)))
        if isinstance(path, Path):
            return PathProvider(Path(os.path.abspath(str(path))))