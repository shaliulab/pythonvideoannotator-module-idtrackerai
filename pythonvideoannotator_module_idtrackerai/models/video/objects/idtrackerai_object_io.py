import time, copy, numpy as np, os, logging
import warnings
from confapp import conf, load_config
import tqdm
try:
    import sys

    sys.path.append(os.getcwd())
    import local_settings # type: ignore

    conf += local_settings
except Exception as e:
    print(e)
    pass

from datetime import datetime
from idtrackerai.utils.py_utils import get_spaced_colors_util
from idtrackerai.tracker.get_trajectories import produce_output_dict
from idtrackerai.tracker.trajectories_to_csv import (
    convert_trajectories_file_to_csv_and_json,
)
from idtrackerai.groundtruth_utils.generate_groundtruth import (
    generate_groundtruth,
)
from idtrackerai.groundtruth_utils.compute_groundtruth_statistics_general import (
    compute_and_save_session_accuracy_wrt_groundtruth,
)

logger = logging.getLogger(__name__)

try:
    import imgstore.constants
    import imgstore.interface
    
    IMGSTORE_ENABLED=True
except ModuleNotFoundError:
    IMGSTORE_ENABLED=False

class IdtrackeraiObjectIO(object):

    FACTORY_FUNCTION = "create_idtrackerai_object"

    def save(self, data={}, obj_path=None):
        idtrackerai_prj_path = os.path.relpath(
            self.idtrackerai_prj_path, obj_path
        )
        data["idtrackerai-project-path"] = idtrackerai_prj_path
        return super().save(data, obj_path)

    def save_updated_identities(self):
        logger.info("Disconnecting list of blobs...")
        self.list_of_blobs.disconnect()

        logger.info("Saving list of blobs...")
        path = os.path.join(
            self.idtrackerai_prj_path,
            "preprocessing",
            "blobs_collection_no_gaps.npy",
        )
        np.save(path, self.list_of_blobs)
        logger.info("List of blobs saved")

        timestamp_str = datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d_%H%M%S"
        )

        trajectories_wo_gaps_file = os.path.join(
            self.idtrackerai_prj_path,
            "trajectories_wo_gaps",
            "trajectories_wo_gaps_{}.npy".format(timestamp_str),
        )
        logger.info("Producing trajectories without gaps ...")
        trajectories_wo_gaps = produce_output_dict(
            self.list_of_blobs.blobs_in_video, self.video_object
        )
        logger.info("Saving trajectories without gaps...")
        np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
        if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
            logger.info("Saving trajectories in csv format...")
            convert_trajectories_file_to_csv_and_json(
                trajectories_wo_gaps_file
            )
        logger.info("Trajectories without gaps saved")

        trajectories_file = os.path.join(
            self.idtrackerai_prj_path,
            "trajectories",
            "trajectories_{}.npy".format(timestamp_str),
        )
        logger.info("Producing trajectories")
        trajectories = produce_output_dict(
            self.list_of_blobs.blobs_in_video, self.video_object
        )
        logger.info("Saving trajectories...")
        np.save(trajectories_file, trajectories)
        if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
            logger.info("Saving trajectories in csv format...")
            convert_trajectories_file_to_csv_and_json(trajectories_file)
        logger.info("Trajectories saved")
        logger.info("Saving video object...")
        self.video_object.save()
        logger.info("Video saved")

    def compute_gt_accuracy(self, generate=True, start=0, end=-1):
        if generate:
            generate_groundtruth(
                self.video_object,
                blobs_in_video=self.list_of_blobs.blobs_in_video,
                start=start,
                end=end,
                save_gt=True,
            )

        compute_and_save_session_accuracy_wrt_groundtruth(
            self.video_object, "normal"
        )
        logger.info(f"{self.video_object.gt_accuracy}")

    def load(self, data, obj_path):
        path = data.get("idtrackerai-project-path", None)
        if path is None:
            return

        idtrackerai_prj_path = os.path.join(obj_path, path)

        logger.info("Loading video object...")
        vidobj_path = os.path.join(idtrackerai_prj_path, "video_object.npy")
        videoobj = np.load(vidobj_path, allow_pickle=True).item()
        videoobj.update_paths(vidobj_path)
        logger.info("Video object loaded")

        self.load_from_idtrackerai(idtrackerai_prj_path, videoobj)

    def load_from_idtrackerai(self, project_path, video_object=None):

        vidobj_path = os.path.join(project_path, "video_object.npy")
        if video_object is None:
            video_object = np.load(vidobj_path, allow_pickle=True).item()
            video_object.update_paths(vidobj_path)

        self.idtrackerai_prj_path = project_path

        logger.info("Updating paths in video object")
        self.video_object = video_object
        video_object.update_paths(vidobj_path)
        logger.info("Paths updated")

        path = os.path.join(
            project_path, "preprocessing", "blobs_collection_no_gaps.npy"
        )
        if not os.path.exists(path):
            path = os.path.join(
                project_path, "preprocessing", "blobs_collection.npy"
            )

        logger.info("Loading list of blobs...")
        self.list_of_blobs = np.load(path, allow_pickle=True).item()
        logger.info("List of blobs loaded")
        logger.info("Connecting list of blobs...")
        self.list_of_blobs.reconnect_from_cache()
        logger.info("List of blobs connected")
        if IMGSTORE_ENABLED: self.align_list_of_blobs()
        logger.info("Loading fragments...")
        path = os.path.join(project_path, "preprocessing", "fragments.npy")
        if (
            not os.path.exists(path)
            and self.video_object.user_defined_parameters["number_of_animals"]
            == 1
        ):
            self.list_of_framents = None
            logger.info("Fragments did not exist")
        else:
            self.list_of_framents = np.load(path, allow_pickle=True).item()
            logger.info("Loading fragments...")
        self.colors = get_spaced_colors_util(
            self.video_object.user_defined_parameters["number_of_animals"],
            black=True,
        )


    @staticmethod
    def find_frame_number(index, ft):
        """
        Given a frame time, return the frame number
        of the first frame after that in the index
        """
        # NOTE:
        # This is more efficient than the commented code below because
        # this skips the "ORDER BY" SQL statement, which is very slow
        # In this way, we dont need the ORDER BY statement because we take
        # advantage of the fact that the frame times are already ordered
        
        # Here we look for the first frame_number in the master store that is ahead of the passed frame time
        # (from the selected or delta store) and we subtract one from it
        # to get the first one in the past instead of the future
        cur = index._conn.cursor()
        cmd="SELECT * from frames WHERE (frame_time - ?) >= 0 LIMIT 1;"
        cur.execute(cmd, (ft,))
        data = cur.fetchone()
        # TODO Check data
        chunk, frame_idx, frame_number, frame_time = data
        return frame_number


    def _align_blobs(self, master_index, frame_numbers_matching, frame_times_matching, frame_numbers_master, start_of_data, end_of_data, frame_min, frame_max):
             
        for i in tqdm.tqdm(frame_numbers_matching[:start_of_data], desc="Generating index for blob alignment ..."):
            frame_numbers_master.append(int(frame_min))

        # TODO
        # # This is parallelizable?
        for ft in tqdm.tqdm(frame_times_matching[start_of_data:end_of_data], desc="Generating index for blob alignment ..."):
            frame_number=self.find_frame_number(master_index, ft)
            target_frame_number = int(max(0, frame_number-1))
            frame_numbers_master.append(target_frame_number)

        for i in tqdm.tqdm(frame_numbers_matching[end_of_data:], desc="Generating index for blob alignment ..."):
            frame_numbers_master.append(int(frame_max))
            
        aligned_blobs=[]
        for frame_number in tqdm.tqdm(frame_numbers_master, desc="Aligning blobs"):
            if len(self.list_of_blobs.blobs_in_video) <= frame_number:
                frame_number = len(self.list_of_blobs.blobs_in_video)
            aligned_blobs.append(self.list_of_blobs.blobs_in_video[frame_number-1])


        return aligned_blobs


    def align_list_of_blobs(self):
        """
        This method takes the blobs_in_video and duplicates its elements
        (blobs_in_frame) so the number of blobs_in_frame matches the
        number of frames of the selected store
        (which is supposed to have a higher framerate)
        """


        config = load_config(imgstore.constants)
        cap=imgstore.interface.VideoCapture(
            self.video_object.video_path,
            chunk=self.video_object._chunk
        )

        if getattr(config, "MULTI_STORE_ENABLED", False):
            cap.select_store(config.SELECTED_STORE)            
        
            frame_numbers_master=[]
            metadata=cap._selected.get_frame_metadata()
            frame_numbers = metadata["frame_number"]
            frame_times = metadata["frame_time"]
            index=cap._master._index

            logger.info("Trimming delta index ")
            frame_min = index._summary("frame_min")
            frame_max = index._summary("frame_max")
            frame_time_min = index._summary("frame_time_min")
            frame_time_max = index._summary("frame_time_max")
            indices = [i for i, ft in enumerate(frame_times) if ft >= frame_time_min and ft <= frame_time_max]
            # we assume the frame_times are sorted in increasing manner
            # the indices indices are also sorted
            frame_times_matching = []
            frame_numbers_matching = []
            
            for i in tqdm.tqdm(range(len(frame_times)), desc="Trimming ..."):
                if i < indices[0]:
                    frame_times_matching.append(frame_time_min)
                    frame_numbers_matching.append(frame_min)
                elif i > indices[-1]:
                    frame_times_matching.append(frame_time_max)
                    frame_numbers_matching.append(frame_max)
                else:
                    frame_times_matching.append(frame_times[i])
                    frame_numbers_matching.append(frame_numbers[i])
                    
            logger.info("Done")
            start_of_data = self.find_frame_number(cap._selected._index, index.get_chunk_metadata(self.video_object._chunk)["frame_time"][0])
            end_of_data = self.find_frame_number(cap._selected._index, index.get_chunk_metadata(self.video_object._chunk)["frame_time"][-1])


            print(f"Starting frame: {start_of_data}, Ending frame {end_of_data}")
            # aligned_blobs = self._align_blobs(index, frame_numbers_matching, frame_times_matching, frame_numbers_master, start_of_data, end_of_data, frame_min, frame_max)


            # crossindex=cap._crossindex.loc[np.bitwise_and(
            #     cap._crossindex[("selected", "frame_time")] >= cap._crossindex.loc[0, ("master", "frame_time")],
            #     cap._crossindex[("selected", "frame_time")] <= cap._crossindex.loc[cap._crossindex.index[-1], ("master", "frame_time")]
            # )]
            
            aligned_blobs=[]
            warned=False
            
            # number_of_frames = cap.crossindex.get_number_of_frames("selected")
            rows = cap.crossindex.get_all_master_fn()

            # for selected_fn in tqdm.tqdm(range(number_of_frames), desc="Aligning blobs"):
            for row in tqdm.tqdm(rows, desc="Aligning blobs"):
                try:
                    # master_fn = cap.crossindex.find_master_fn(selected_fn)
                    master_fn=row[0]
                    aligned_blobs.append(self.list_of_blobs.blobs_in_video[master_fn])
                except IndexError:
                    # TODO
                    # Figure out why the crossindex has a number of rows equal to frame_times (expected)
                    # but the highest frame number is the length of the original blobs in video,
                    # which is unexpected since it should be that - 1 (because the frame numbers are 0 indexed)
                    aligned_blobs.append([])
                    if not warned:
                        warnings.warn("Some frames from the index have been ignored", stacklevel=2)
                        warned=True

            assert len(aligned_blobs) == len(frame_times)
            self.list_of_blobs.blobs_in_video = aligned_blobs

        else:
            pass

