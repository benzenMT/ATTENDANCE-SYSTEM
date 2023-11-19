import { getCookie } from "../cookies";
import "./layout.css";
import { Link } from "react-router-dom";
function DefaultLayout({ children }) {
  return (
    <div className="layout">
      <div className="layout_navbar">
        <Link to="/" className="layout_navbar_item">
          <i class="fa-solid fa-house icon_home"></i>
        </Link>
        <Link to="/attendance" className="layout_navbar_item">
          <i class="fa-solid fa-clipboard-user icon_home"></i>
        </Link>
        {getCookie("token") && (
          <>
            <Link to="/camera" className="layout_navbar_item">
              <i class="fa-solid fa-camera-retro icon_home"></i>
            </Link>
          </>
        )}
        {getCookie("token") && (
          <>
            <Link to="/manager" className="layout_navbar_item">
              <i class="fa-solid fa-list-check icon_home"></i>
            </Link>
          </>
        )}
      </div>
      <div
        style={{
          paddingTop: "",
        }}
        className="layout_container"
      >
        {children}
      </div>
    </div>
  );
}

export default DefaultLayout;
