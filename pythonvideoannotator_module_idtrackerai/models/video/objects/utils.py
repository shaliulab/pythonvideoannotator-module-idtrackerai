import os.path

from confapp import conf
def get_imgstore_path(project_path):
    if getattr(conf, "FLYHOSTEL_DIRECTORY_VERSION", 2) == 2:
        imgstore_path = os.path.join(project_path, "..", "..", "metadata.yaml")
    
    elif getattr(conf, "FLYHOSTEL_DIRECTORY_VERSION", 2) == 1:
        imgstore_path = os.path.join(project_path, "..", "metadata.yaml")

    return imgstore_path


def get_chunk_numbers(idtracker_videoobj):
    try:
        # NOTE
        # This should work on all future idtrackerai analysis runs using imgstore
        chunk_numbers = idtracker_videoobj._chunk_numbers
    except:
        # for now this allows me to figure it out
        session = os.path.dirname(idtracker_videoobj.path_to_video_object)
        chunk = int(session.split("_")[1])
        chunk_numbers = [chunk-1, chunk, chunk+1]
        chunk_numbers = [e for e in chunk_numbers if e >= 0]
        idtracker_videoobj._chunk_numbers = chunk_numbers
    
    return chunk_numbers

def get_chunk(idtracker_videoobj):
    try:
        chunk = idtracker_videoobj._chunk
    except:
        session = os.path.dirname(idtracker_videoobj.path_to_video_object)
        chunk = int(session.split("_")[1])
        idtracker_videoobj._chunk = chunk

    
    return chunk  
    