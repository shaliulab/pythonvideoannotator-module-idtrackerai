import time, copy, numpy as np, os, logging
import pandas as pd
from pythonvideoannotator_module_idtrackerai.models.video.objects.utils import get_chunk_numbers, get_chunk, get_imgstore_path
from tqdm import tqdm
from confapp import conf
import pandas as pd


try:
    import sys

    sys.path.append(os.getcwd())
    import local_settings # type ignore

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

from idtrackerai.list_of_blobs import ListOfBlobs

logger = logging.getLogger(__name__)



class IdtrackeraiObjectIO(object):

    FACTORY_FUNCTION = "create_idtrackerai_object"

    def save(self, data={}, obj_path=None):
        idtrackerai_prj_path = os.path.relpath(
            self.idtrackerai_prj_path, obj_path
        )
        data["idtrackerai-project-path"] = idtrackerai_prj_path
        return super().save(data, obj_path)

    def undo_multicamera_integration(self, delimitations):
        start_time = time.time()
        
        self.list_of_blobs.blobs_in_video = self.list_of_blobs.blobs_in_video[
            delimitations[0]:delimitations[1]
        ]
        frame_index = pd.DataFrame(self.list_of_blobs.time_index_df["frame_number"]).drop_duplicates()
        
        self.list_of_blobs.blobs_in_video = [
            self.list_of_blobs.blobs_in_video[i] for i in frame_index.index
        ]
        
        
        for i, blobs_in_frame in enumerate(self.list_of_blobs.blobs_in_video):
            for blob in blobs_in_frame:
                blob._interpolated_frame_number = blob.frame_number
                # blob.frame_number = blob._frame_number_in_original_video
                blob.frame_number = i
        end_time = time.time()
        print(f"Undid changes in {end_time - start_time}")
 
    def redo_multicamera_integration(self, delimitations):

        start_time = time.time()
        
        integrated_blobs = [
            self.list_of_blobs.blobs_in_video[i]
            for i in self.list_of_blobs.time_index_df["frame_number"] 
        ]

        self.list_of_blobs.blobs_in_video = [[],] * delimitations[0] + \
            integrated_blobs + \
            [[],] * (delimitations[2] - delimitations[1])
        
        for blobs_in_frame in self.list_of_blobs.blobs_in_video:
            for blob in blobs_in_frame:
                blob._frame_number_in_original_video = blob.frame_number
                blob.frame_number = blob._interpolated_frame_number

        end_time = time.time()
        print(f"Redid changes in {end_time - start_time}")


    def save_updated_identities(self):
        # undo the changes I did to show multicamera feed
        delimitations = getattr(self.list_of_blobs, "_delimitations", False)
        if delimitations:
            self.undo_multicamera_integration(delimitations)

    
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
        # import ipdb; ipdb.set_trace()
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
        if delimitations:
            self.redo_multicamera_integration(delimitations)
        

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

    def load_from_idtrackerai(self, project_path, video_object=None, backend="cv2"):

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
        self.list_of_blobs = ListOfBlobs.load(path)       
        logger.info("List of blobs loaded")
        logger.info("Connecting list of blobs...")
        if not conf.RECONNECT_BLOBS_FROM_CACHE:
            self.list_of_blobs.compute_overlapping_between_subsequent_frames()
        else:
            self.list_of_blobs.reconnect_from_cache()

        logger.info("List of blobs connected")

        self.list_of_blobs.blobs_in_video = self._interpolate_blobs(self.list_of_blobs, backend=backend)

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

        
    def _interpolate_blobs(self, list_of_blobs, backend="imgstore"):

        print("Interpolating blobs to adjust for multicamera feed")

        if backend=="imgstore":

            chunk = self.video._videocap._chunk

            if self._blobs_in_video is None:

                self._chunk = chunk
                # frames_per_chunk = len(self.video._videocap._delta_time_generator._chunk_md["frame_number"])
                metadata = self.video._videocap._delta_time_generator.get_frame_metadata()

                frame_index_all = metadata["frame_number"]
                time_index_all = metadata["frame_time"]

                blobs_in_video = list_of_blobs.blobs_in_video
                
                main_time = self.video._videocap._main_store._get_chunk_metadata(
                    self.video._videocap._main_store._chunk
                )["frame_time"]
                    
                interval = (min(main_time), max(main_time))

                time_index = [e for e in time_index_all if e >= interval[0] and e <= interval[1]]

                interpolated_blobs = []

                time_index_df = pd.DataFrame({"time": time_index})
                main_time_df = pd.DataFrame({"time": main_time, "frame_number": list(range(len(blobs_in_video)))})
                time_index_df = pd.merge_asof(time_index_df, main_time_df, direction="backward", on="time") 
                time_index_df.to_csv("/tmp/session_index.csv")
                list_of_blobs.time_index_df = time_index_df
                interpolated_blobs = [blobs_in_video[i] for i in time_index_df["frame_number"]]
                
                index_of_first_frame = frame_index_all[
                    np.where(time_index_df.head(1)["time"].values == np.array(time_index_all))[0].tolist()[0]
                ]
                index_of_last_frame = frame_index_all[
                    np.where(time_index_df.tail(1)["time"].values == np.array(time_index_all))[0].tolist()[0]
                ]



                self._previous_frames = [[], ] * index_of_first_frame
                extended_blobs_in_video = self._previous_frames + interpolated_blobs
                # TODO Is this = index_of_last_frame? or index_of_last_frame - 1?
                end_of_frames_with_blob = len(extended_blobs_in_video)
                frames_after_last_frame_with_blob = (len(frame_index_all) - index_of_last_frame)

                list_of_blobs._delimitations = (
                    index_of_first_frame,
                    end_of_frames_with_blob,
                    end_of_frames_with_blob + frames_after_last_frame_with_blob
                )
                
                for i in tqdm(
                    range(index_of_first_frame, len(extended_blobs_in_video)),
                    desc="Adjusting frame number of blobs"
                ):

                    blobs = extended_blobs_in_video[i]
                    for blob in blobs:
                        blob._frame_number_in_original_video = blob.frame_number
                        blob.frame_number = i
                    # if i == index_of_first_frame:
                    #     import ipdb; ipdb.set_trace()

                next_frames = [[],] * frames_after_last_frame_with_blob
                extended_blobs_in_video += next_frames
                self._blobs_in_video_original = self._blobs_in_video
                self._blobs_in_video = extended_blobs_in_video
                return self._blobs_in_video
            else:
                return self._blobs_in_video
                
        elif backend=="cv2":
            return list_of_blobs.blobs_in_video

        else:
            logger.warning(f"Unknown backend {backend}")
            return list_of_blobs.blobs_in_video
