import "./Register.css";
import React, { useState } from "react";
import Modal from "react-modal";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { URL } from "../../link_server";
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
function Register() {
  const notify = (data) => toast(`${data}`);
  const [register, setRegister] = useState({
    Managername: "",
    Managerpass: "",
  });
  const [checkpass, setCheckpass] = useState(true);
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
  }
  const handleRegister = () => {
    if (!register.Managername || !register.Managerpass) {
      alert("ĐIỀN ĐỦ THÔNG TIN ĐI BẠN ƠI");
      return;
    }
    const data = {
      adminName: register.Managername,
      adminPass: register.Managerpass,
    };
    axios
      .post(`${URL}/admin/register`, data)
      .then(function (response) {
        console.log(response);
        notify(response.data.message);
        if (response.data.errCode === 0) {
          setTimeout(closeModal, 2000);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
  };
  return (
    <>
      <div className="register" onClick={openModal}>
        <i
          class="fa-solid fa-right-to-bracket"
          style={{ padding: "0px 5px" }}
        ></i>
        Register
      </div>
      <div>
        <Modal
          isOpen={modalIsOpen}
          onAfterOpen={afterOpenModal}
          onRequestClose={closeModal}
          style={customStyles}
          contentLabel="Example Modal"
        >
          <div>
            <div ref={(_subtitle) => (subtitle = _subtitle)}></div>
            <div className="modal_register">
              <h3 style={{ paddingBottom: "10px" }}>Register</h3>
              <h4>
                ManagerName : <i class="fa-solid fa-signature"></i>
              </h4>

              <input
                onChange={(e) => {
                  setRegister({ ...register, Managername: e.target.value });
                }}
              />
              <br />
              <h4>
                ManagerPass : <i class="fa-solid fa-lock"></i>
              </h4>
              <input
                type={checkpass ? "password" : "text"}
                onChange={(e) => {
                  setRegister({ ...register, Managerpass: e.target.value });
                }}
              />
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
              <div className="btn_register" onClick={handleRegister}>
                Register
              </div>
            </div>
          </div>
        </Modal>
      </div>
      <ToastContainer />
    </>
  );
}
export default Register;
