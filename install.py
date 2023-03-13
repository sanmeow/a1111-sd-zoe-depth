import launch

if not launch.is_installed("trimesh"):
    try:
        launch.run_pip("install trimesh", "requirements for trimesh")
    except:
        print("Can't install trimesh. Please follow the readme to install manually")