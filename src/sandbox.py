from visualization import PointCloudVisualizer

def main():
    pcv = PointCloudVisualizer("./data/Bunny.e57")
    pcv.visualize()

    pcv = PointCloudVisualizer("./data/StSulpice-Cloud-50mm.e57")
    pcv.visualize()

if __name__ == "__main__":
    main()