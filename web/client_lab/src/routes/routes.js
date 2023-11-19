import Home from "../pages/Home/Home";
import Camera from "../pages/Camera/Camera";
import Manager from "../pages/Manager/Manage";
import Attendance from "../pages/Attendance/Attendance";

const publicRoutes = [
  { path: "/", Component: Home },
  { path: "/camera", Component: Camera },
  { path: "/manager", Component: Manager },
  { path: "/attendance", Component: Attendance },
];

export { publicRoutes };
