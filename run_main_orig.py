# run_main.py 수정안: scene_path + object_path 두 디렉토리에서 ply를 로드

import click
from pathlib import Path
from splatviz import Splatviz
import time

@click.command()
@click.option("--data_path", default="./resources/sample_scenes", help="root path for .ply files", metavar="PATH")
@click.option("--scene_path", default="", help="Path to a single scene .ply file", type=click.Path(exists=True))
@click.option("--objects_path", default="", help="Path to a directory of object .ply files", type=click.Path(exists=True))
@click.option("--mode", default="default", type=str, help="viewer mode")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
@click.option("--gan_path", default=None)
def main(data_path, scene_path, objects_path, mode, host, port, gan_path):
    # If scene_path is provided, use it and any .ply files in objects_path
    if scene_path:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port,
                             scene_path=scene_path, objects_path=objects_path)
    else:
        splatviz = Splatviz(data_path=data_path, mode=mode, host=host, port=port)
    while not splatviz.should_close():
        splatviz.draw_frame()
        time.sleep(1/60)
    splatviz.close()

if __name__ == '__main__':
    main()