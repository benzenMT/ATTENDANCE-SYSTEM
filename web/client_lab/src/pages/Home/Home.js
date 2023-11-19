import Register from "../../components/Register/Register";
import React, { useState, useEffect } from "react";
import Modal from "react-modal";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { URL } from "../../link_server";
import "./Home.css";
import { setCookie, getCookie, deleteAllCookies } from "../../cookies";
const customStyles = {
  content: {
    top: "50%",
    left: "50%",
    right: "auto",
    bottom: "auto",
    marginRight: "-50%",
    transform: "translate(-50%, -50%)",
    boxShadow: "rgba(3, 102, 214, 0.3) 0px 0px 0px 3px",
  },
};

function Home(props) {
  const [checked, setChecked] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [checkeduser, setCheckeduser] = useState(true);
  const [checkpass, setCheckpass] = useState(true);
  const [datalogin, setDatalogin] = useState({
    managerName: "",
    managerPass: "",
  });
  //modal
  let subtitle;
  const [modalIsOpen, setIsOpen] = React.useState(false);
  function openModal() {
    setIsOpen(true);
  }
  function afterOpenModal() {
    subtitle.style.color = "#333";
  }
  function closeModal() {
    setIsOpen(false);
    console.log("data : ", checkeduser);
    if (checkeduser) {
      setChecked(false);
    } else {
      setChecked(true);
    }
  }
  //modal
  //send data to parent
  const sendDataToParent = () => {
    props.onDataFromChild(true);
  };

  const handleLogin = () => {
    const notify = (data) => toast(`${data}`);
    if (!datalogin.managerName || !datalogin.managerPass) {
      alert("NHẬP ĐỦ THÔNG TIN ĐI BẠN ƠI");
      return;
    }
    const data = {
      adminName: datalogin.managerName,
      adminPass: datalogin.managerPass,
    };
    axios
      .post(`${URL}/admin/login`, data)
      .then(function (response) {
        console.log(response);
        notify(response.data.message);
        if (response.data.errCode === 0) {
          sendDataToParent();
          setCookie("token", response.data.token, 1);
          closeModal();
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };
  useEffect(() => {
    if (getCookie("token")) {
      setChecked(true);
      setCheckeduser(false);
    }
  }, [modalIsOpen]);
  return (
    <>
      <div className="home">
        <div className="home_manager">
          <div className="home_manager_item">
            <h3>Manager</h3>
            <div className="manager_img">
              <img
                className="manager_img_img"
                alt=""
                src="https://cdn1.iconfinder.com/data/icons/digital-strategy-1/64/manager-business-man-avatar-512.png"
              />
            </div>
            <div className="manager_check">
              <label className="switch">
                <input
                  type="checkbox"
                  checked={checked}
                  onClick={() => {
                    if (getCookie("token")) {
                      return;
                    }
                    setChecked(true);
                    openModal();
                  }}
                />
                <span className="slider round"></span>
              </label>
            </div>
          </div>
          <div className="home_manager_item">
            <h3>User</h3>
            <div className="manager_img">
              <img
                width="100%"
                className="manager_img_img"
                alt=""
                src="https://img.freepik.com/free-vector/illustration-user-avatar-icon_53876-5907.jpg"
              />
            </div>
            <div className="manager_check">
              <label className="switch">
                <input type="checkbox" checked={checkeduser} />
                <span className="slider round"></span>
              </label>
            </div>
          </div>
          <div className="home_text">
            <h3>
              <i
                class="fa-solid fa-clipboard"
                style={{ padding: "2px 10px" }}
              ></i>
              Attendance Tracking System
            </h3>
            <p>
              An attendance tracking system is a vital tool used in various
              organizations <br /> and educational institutions to monitor and
              record the presence of individuals
              <br /> at a specific location or event. This system is designed to
              streamline the process
              <br /> of taking attendance, making it more efficient and accurate
            </p>
          </div>
        </div>
        <div className="home_img">
          <img
            style={{ width: "100%" }}
            alt=""
            src="https://firebasestorage.googleapis.com/v0/b/imagepostcommentlike.appspot.com/o/395098978_842909694291854_5057374663840811065_n.jpg?alt=media&token=199d643c-ec68-46e5-999d-0fed7814ff21&_gl=1*1dkwt4h*_ga*MTg1MjYxMzE0MC4xNjk4MDgxMjIw*_ga_CW55HF8NVT*MTY5ODY4MzEyNS45LjEuMTY5ODY4MzIwMS41Mi4wLjA."
          />
        </div>
        {getCookie("token") ? (
          <>
            <div className="logout" onClick={deleteAllCookies}>
              Log Out
            </div>
          </>
        ) : (
          <Register></Register>
        )}
      </div>

      <Modal
        isOpen={modalIsOpen}
        onAfterOpen={afterOpenModal}
        onRequestClose={closeModal}
        style={customStyles}
        contentLabel="Example Modal"
      >
        <div>
          <div ref={(_subtitle) => (subtitle = _subtitle)}></div>
          <div className="modal_login">
            <h3>Login</h3>
            <h4>
              ManagerName : <i class="fa-solid fa-signature"></i>
            </h4>
            <input
              onChange={(e) => {
                setDatalogin({ ...datalogin, managerName: e.target.value });
              }}
            />
            <h4>
              ManagerPass : <i class="fa-solid fa-unlock"></i>
            </h4>
            <div
              className="icon_eyes"
              onClick={() => {
                setCheckpass(!checkpass);
              }}
            >
              <>
                {checkpass ? (
                  <>
                    <i class="fa-solid fa-eye"></i>
                  </>
                ) : (
                  <>
                    <i class="fa-solid fa-eye-slash"></i>
                  </>
                )}
              </>
            </div>
            <input
              type={checkpass ? "password" : "text"}
              onChange={(e) => {
                setDatalogin({ ...datalogin, managerPass: e.target.value });
              }}
            />
            <div className="btn_login" onClick={handleLogin}>
              Login
            </div>
          </div>
        </div>
      </Modal>
      <ToastContainer></ToastContainer>
    </>
  );
}

export default Home;
