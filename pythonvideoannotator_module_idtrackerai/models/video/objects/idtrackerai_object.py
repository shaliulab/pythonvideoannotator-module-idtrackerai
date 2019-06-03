import numpy as np, cv2

from AnyQt import QtCore
from confapp import conf
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlText
from pyforms.controls import ControlButton
from pyforms.controls import ControlCheckBox
from .idtrackerai_object_io import IdtrackeraiObjectIO
from .idtrackerai_object_mouse_events import IdtrackeraiObjectMouseEvents
from idtrackerai.postprocessing.assign_them_all import close_trajectories_gaps
from pythonvideoannotator_models.models.video.objects.video_object import VideoObject
from pythonvideoannotator_models_gui.models.video.objects.object2d.datasets.dataset_gui import DatasetGUI




class SelectedBlob(object):
    """
    Class to store information about the selected blob.
    """
    def __init__(self, blob, blob_id, blob_pos):
        """
        :param int blob: Blob object.
        :param int blob_id: Id of the blob object.
        :param int blob_pos: Blob position.
        """
        self.blob     = blob
        self.position = blob_pos
        self.identity = blob_id




class IdtrackeraiObject(IdtrackeraiObjectMouseEvents, DatasetGUI, VideoObject, IdtrackeraiObjectIO, BaseWidget):

    def __init__(self, video):

        self._nametxt = ControlText('Name', default='Untitled')
        self._closepaths_btn = ControlButton('Interpolate trajectories', default=self.__close_trajectories_gaps)
        self._del_centroids_btn = ControlButton('Delete centroids', default=self.__delete_centroids_btn_evt)
        #self._add_blob_ondblclick = ControlCheckBox('Add blob on double click', default=False)

        DatasetGUI.__init__(self)
        VideoObject.__init__(self)
        IdtrackeraiObjectMouseEvents.__init__(self)
        BaseWidget.__init__(self, 'IdtrackerAi Object', parent_win=video)

        self._video = video
        self._video += self
        self.create_tree_nodes()

        self.video_object     = None # IdtrackerAI VideoObject
        self.list_of_blobs    = None # IdtrackerAI ListOfBlobs
        self.list_of_framents = None # IdtrackerAI ListOfFragments


        self.formset = [
            '_nametxt',
            #'_add_blob_ondblclick',
            '_del_centroids_btn',
            '_closepaths_btn',
            ' '
        ]

    def create_tree_nodes(self):
        self.treenode = self.tree.create_child(self.name, icon=conf.ANNOTATOR_ICON_OBJECT, parent=self.video.treenode)
        self.treenode.win = self

    ######################################################################
    ### Events events ####################################################
    ######################################################################

    def __delete_centroids_btn_evt(self):
        if self.selected is None:
            return

        frame_index = self.mainwindow.timeline.value
        blobs = self.list_of_blobs.blobs_in_video[frame_index]
        index = blobs.index(self.selected.blob)
        if index>=0:
            blobs.pop(index)

    def __close_trajectories_gaps(self):
        """
        Make the interpolations.
        :return:
        """
        self.list_of_blobs = close_trajectories_gaps(
            self.video_object,
            self.list_of_blobs,
            self.list_of_framents
        )



    ######################################################################
    ### Key events #######################################################
    ######################################################################

    def key_release_event(self, evt):

        # Jump to the next crossing.
        if evt.key() == QtCore.Qt.Key_W:
            self.__jump2next_crossing()

        # Jump to the previous crossing.
        elif evt.key() == QtCore.Qt.Key_S:
            self.__jump2previous_crossing()


    def __jump2previous_crossing(self):
        """
        Jump to previous crossing.
        :return:
        """
        frame_index = self.mainwindow.timeline.value
        frames = self.list_of_blobs.blobs_in_video

        for i in range(frame_index-1, -1, -1):
            for blob in frames[i]:
                if blob.is_a_crossing:
                    self.mainwindow.timeline.value = blob.frame_number
                    return

    def __jump2next_crossing(self):
        """
        Jump to the next crossing
        :return:
        """
        frame_index = self.mainwindow.timeline.value
        frames = self.list_of_blobs.blobs_in_video

        for i in range(frame_index+1, len(frames) ):
            for blob in frames[i]:
                if blob.is_a_crossing:
                    self.mainwindow.timeline.value = blob.frame_number
                    return


    ######################################################################
    ### PROPERTIES #######################################################
    ######################################################################

    def draw(self, frame, frame_index):
        """
        Draw the objects in the current frame of the video.
        :param numpy.array frame: Image of the current frame.
        :param int frame_index: Index of the current frame in the video.
        :return numpy.array: The frame to draw.
        """
        if self.list_of_blobs is None: return

        image = frame.copy()
        blobs = self.list_of_blobs.blobs_in_video[frame_index]

        for blob in blobs:

            identities = blob.final_identity if isinstance(blob.final_identity, list) else [blob.final_identity]
            centroids  = blob.interpolated_centroids if hasattr(blob, 'interpolated_centroids') else [blob.centroid]
            contour    = blob.contour

            for identity, centroid in zip(identities, centroids):

                pos   = int(round(centroid[0],0)), int(round(centroid[1],0))
                color = self.colors[identity] if identity is not None else self.colors[0]

                cv2.polylines(frame, np.array([contour]), True, (0, 255, 0), 1)

                cv2.circle(image, pos, 8, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(image, pos, 6, color, -1, lineType=cv2.LINE_AA)

                if self.selected and self.selected.identity==identity:
                    cv2.circle(image, pos, 10, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                if identity is not None:

                    if blob.user_generated_identity is not None:
                        idroot = 'u-'
                    elif blob.identity_corrected_closing_gaps is not None and not blob.is_an_individual:
                        idroot = 'c-'
                    else:
                        idroot = ''

                    idstr      = idroot + str(identity)
                    text_size  = cv2.getTextSize(idstr, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness=2)
                    text_width = text_size[0][0]
                    str_pos    = pos[0] - text_width // 2, pos[1] - 12

                    cv2.putText(image, idstr, str_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(image, idstr, str_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2, lineType=cv2.LINE_AA)
                else:
                    bounding_box = blob.bounding_box_in_frame_coordinates
                    rect_color = blob.rect_color if hasattr(blob, 'rect_color') else (255,0,0)
                    cv2.rectangle(image, bounding_box[0], bounding_box[1], rect_color, 2)


        if self._tmp_object_pos:
            cv2.circle(image, self._tmp_object_pos, 8, (50,50,50), 2, lineType=cv2.LINE_AA)

        # Draw with transparency.
        cv2.addWeighted(frame, 0.5, image, 0.5, 0, dst=frame)

    ######################################################################
    ### PROPERTIES #######################################################
    ######################################################################

    @property
    def name(self):
        return self._nametxt.value

    @property
    def mainwindow(self): return self.video.mainwindow

    @property
    def tree(self): return self.video.tree

    @property
    def video_capture(self): return self.video.video_capture

    @property
    def parent_treenode(self):  return self.video.treenode
