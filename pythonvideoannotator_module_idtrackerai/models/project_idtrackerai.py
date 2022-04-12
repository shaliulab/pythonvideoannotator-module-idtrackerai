import os, numpy as np
import logging

logger = logging.getLogger(__name__)

from pythonvideoannotator_module_idtrackerai.models.video.objects.utils import get_chunk_numbers, get_chunk, get_imgstore_path

class IdTrackerProject(object):

    @staticmethod
    def set_filepath(video, idtracker_videoobj, video_path, imgstore_path):
        if os.path.exists(imgstore_path):
            chunk_numbers = get_chunk_numbers(idtracker_videoobj)
            ref_chunk = get_chunk(idtracker_videoobj)

            video.filepath_setter(
                imgstore_path,
                ref_chunk=ref_chunk,
                chunk_numbers=chunk_numbers
            )
            return "imgstore"

        elif os.path.exists(video_path):
            video.filepath_setter(video_path)
            return "cv2"
        else:
            raise Exception("Video not found")

    def load(self, data, project_path=None):
        """
        Check if the the path includes an idtrackerai project, if so load it.
        :param data:
        :param project_path:
        :return:
        """

        # Only if a video annotator project exists in the project
        if os.path.exists(os.path.join(project_path, 'project.json')):
            res = super().load(data, project_path)

            if len(self.videos)==1:
                # expand and select the tree nodes
                self.videos[0].treenode.setExpanded(True)

                if len(self.videos[0].objects)==1:
                    self.videos[0].objects[0].treenode.setSelected(True)
                    self.mainwindow.player.video_index = self.videos[0].objects[0].get_first_frame()

                self.mainwindow.player.call_next_frame()

            return res
        else:
            # Load an idtracker project

            blobs_path  = os.path.join(project_path, 'preprocessing', 'blobs_collection_no_gaps.npy')
            if not os.path.exists(blobs_path):
                blobs_path = os.path.join(project_path, 'preprocessing', 'blobs_collection.npy')


            vidobj_path = os.path.join(project_path, 'video_object.npy')

            if os.path.exists(blobs_path) and os.path.exists(vidobj_path):

                idtracker_videoobj = np.load(vidobj_path, allow_pickle=True).item()

                video = self.create_video()
                video.multiple_files = idtracker_videoobj.open_multiple_files

                video_path = os.path.join(project_path, '..', os.path.basename(idtracker_videoobj._video_path) )
                imgstore_path = get_imgstore_path(project_path)
                backend = self.set_filepath(video, idtracker_videoobj, video_path, imgstore_path)

                obj = video.create_idtrackerai_object()
                obj.load_from_idtrackerai(project_path, idtracker_videoobj, backend=backend)

                # expand and select the tree nodes
                video.treenode.setExpanded(True)
                obj.treenode.setSelected(True)

                first_frame = obj.get_first_frame()
                logger.warning(f"First frame: {first_frame}")
                self.mainwindow.player.video_index = first_frame
                self.mainwindow.player.call_next_frame()
