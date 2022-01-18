import os.path
import math, logging, traceback
from collections import namedtuple
import itertools
Modification = namedtuple("Modification", "session_folder frame_number in_frame_index identity new_blob_identity x y")

logger = logging.getLogger(__name__)


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
        self.blob = blob
        self.position = blob_pos[0], blob_pos[1]
        self.identity = blob_id


class IdtrackeraiObjectMouseEvents(object):

    RADIUS_TO_SELECT_BLOB = 10
    HISTORY_PATH = "python-video-annotator-human_mods.csv"

    def __init__(self):

        self._tmp_object_pos = (
            None  # used to store the temporary object position on drag
        )
        self.selected = None  # Store information about the the selected blob.
        self._drag_active = True
        self._history = []
        # if getattr(self.video_object, "_store", False):
        #     self._blobs_in_video = self._interpolate_blobs()
        
        # else:
        #     self._blobs_in_video = self.list_of_blobs.blobs_in_video
    

    def get_frame_number(self, blob):
        if getattr(self.video_object, "_store", False):
            frame_number = blob.frame_number + len(self._previous_frames)
        else:
            frame_number = blob.frame_number

    def on_click(self, event, x, y):
        """
        Event on click, it is used to select a blob.
        :param event: Click event.
        :param int x: X coordinate.
        :param int y: Y coordinate.
        """

        try:

            frame_index = self.mainwindow.timeline.value

            if not self._add_centroidchk.value:

                selected = False

                p0 = x, y

                # no blobs to select, exit the function
                if not self.list_of_blobs.blobs_in_video[frame_index]:
                    self.selected = None
                    return

                # blobs in the current frame
                blobs = self.list_of_blobs.blobs_in_video[frame_index]

                for blob in blobs:

                    for identity, p1 in zip(
                        blob.final_identities, blob.final_centroids_full_resolution
                    ):

                        # check if which blob was selected
                        if (
                            math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
                            < self.RADIUS_TO_SELECT_BLOB
                        ):

                            self.mainwindow.player.stop()

                            self.selected = SelectedBlob(blob, identity, p1)
                            selected = True
                            break

                if not selected:
                    self.selected = None

            elif self.selected:

                frame_number = self.get_frame_number(self.selected.blob)
                
                if frame_number == frame_index:

                    # ask the new blob identity
                    identity = self.input_int(
                        "Type in the centroid identity", title="Identity", default=1
                    )

                    if not (identity in ["", None]):
                        try:
                            self.selected.blob.add_centroid(
                                self.video_object, (x, y), identity
                            )
                        except Exception as e:
                            logger.debug(str(e), exc_info=True)
                            self.warning(str(e), "Error")

                    self._add_centroidchk.value = False
                    self._tmp_object_pos = None
                    self._drag_active = False

                elif frame_number != frame_index:
                    self.selected = None
                    self._add_centroidchk.value = False
        
        except Exception as error:
            logger.error(error)
            logger.error(traceback.print_exc())
        

    def _save_history(self):

        with open(self.HISTORY_PATH, "w") as fh:

            fh.write(
                "frame_number,in_frame_index,identity,new_blob_identity,x,y\n"
            )
            for row in self._history:
                frame_number = getattr(row, "frame_number_generic", row.frame_number)


                fh.write(f"{row.session_folder},{frame_number},{row.in_frame_index},"\
                         f"{row.identity},{row.new_blob_identity},"\
                         f"{row.x},{row.y}\n")

    def _update_history(self, blob, identity, new_blob_identity, centroid):
        self._history.append(
            Modification(
                os.path.basename(self.video_object._session_folder),
                blob.frame_number_generic,
                blob.in_frame_index,
                identity,
                new_blob_identity,
                *centroid,
            )
        )
        self._save_history()
        return

    def on_double_click(self, event, x, y):

        p0 = x, y
        frame_index = self.mainwindow.timeline.value

        # if one object is selected
        if self.selected is not None:

            identity = self.selected.identity
            blob = self.selected.blob
            centroid = self.selected.position

            # ask the new blob identity
            new_blob_identity = self.input_int(
                "Type in the new identity",
                title="New identity",
                default=identity if identity is not None else 0,
            )

            # Update only if the new identity is different from the old one.
            if new_blob_identity is not None:
                try:

                    blob.update_identity(identity, new_blob_identity, centroid)
                    direction = self._propagate_direction.value
                    if direction in ["all", "future", "past"]:
                        blob.propagate_identity(
                            identity, new_blob_identity, centroid, direction=direction
                        )
                    else:
                        logger.warning("Please pass a supported direction. Either all, future, or past")

                    self._update_history(
                        blob, identity, new_blob_identity, centroid
                    )
                    # log this to a csv file

                except Exception as e:
                    logger.debug(str(e), exc_info=True)
                    self.warning(str(e), "Error")

        elif self._add_blobchk.value:
            new_blob_identity = self.input_int(
                "Type the identity for the new blob",
                title="New identity",
                default=0,
            )
            if not (new_blob_identity in ["", None]):
                try:
                    blob = self.list_of_blobs.add_blob(
                        self.video_object,
                        frame_index,
                        (x, y),
                        new_blob_identity,
                    )
                    # log this to a csv file
                    self._update_history(blob, None, new_blob_identity, (x, y))
                except Exception as e:
                    logger.debug(str(e), exc_info=True)
                    self.warning(str(e), "Error")

    def on_drag(self, p1, p2):

        if (
            self.selected
            and self.selected.identity
            and math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) > 5
        ):

            self._tmp_object_pos = int(round(p2[0], 0)), int(round(p2[1], 0))

    def on_end_drag(self, p1, p2):

        if self._tmp_object_pos and self.selected and self.selected.blob:

            self.selected.blob.update_centroid(
                self.video_object,
                self.selected.position,
                p2,
                self.selected.identity,
            )
            self.selected.position = p2

        self._tmp_object_pos = None
