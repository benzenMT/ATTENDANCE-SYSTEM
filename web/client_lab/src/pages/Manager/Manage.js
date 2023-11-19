import axios from "axios";
import { useEffect, useState } from "react";
import Boxuser from "../../components/Boxuser/Boxuser";
import { URL } from "../../link_server";
import "./Manager.css";
const moment = require("moment");
const now = moment();
const formattedDate = now.format("DD/MM/YYYY");
function Manager() {
  const [fullUser, setFullUser] = useState([]);
  const [userSearch , setUserSearch] = useState({});
  const [checkSearch, setCheckSearch] = useState(false);
    const [search, setSearch] = useState("");
  const searchData = () => {
    const abc = fullUser.find((item) => {
      return item.userCode === search;
    });
    if (abc) {
      setUserSearch(abc);
      setCheckSearch(true);
    } else {
      setCheckSearch(false);
      return;
    }
  };

const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      searchData();
    }
  }


  useEffect(() => {
    axios
      .get(`${URL}/user/full`, {
        withCredentials: true,
      })
      .then(function (response) {
        if (response.data.data.length > 0) {
          setFullUser([...response.data.data]);
        }
      })
      .catch(function (error) {
        console.log(error);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <div className=" manager ">
        <div className="manager_user">
          <div className="manager_user_time">
            <i class="fa-regular fa-calendar-days"></i>
            {formattedDate} (Day/Month/Year)
          </div>
          <div className="manager_user_input" style={{ position: "relative" }}>
            <i class="fa-solid fa-magnifying-glass" onClick={searchData}></i>
            <input
              value={search}
              placeholder="Ex : 20203520"
              onChange={(e) => {
                setSearch(e.target.value);
              }}
              onKeyDown={handleKeyDown}
            />
            {userSearch && checkSearch && (
              <div className="manager_user_box_search">
                <Boxuser key={userSearch.id} check={false} infor={userSearch} />
                <div
                  className="box_search_close"
                  onClick={() => {
                    setSearch("");
                    setCheckSearch(false);
                  }}
                >
                  <i
                    class="fa-solid fa-xmark"
                    style={{ padding: "12px 0px 0px 0px" }}
                  ></i>
                </div>
              </div>
            )}
          </div>
          <div className="manager_user_img">
            <img
              width={"80%"}
              alt=""
              src="https://blog.cdn.cmarix.com/blog/wp-content/uploads/2022/05/Attendance-Management-System.png"
            />
          </div>
        </div>
        <div className="manager_list">
          <h3 className="manager_list_name">
            <i class="fa-solid fa-user-group"></i> List User
          </h3>
          <div className="manager_list_user">
            {fullUser &&
              fullUser.length > 0 &&
              fullUser.map((item, index) => {
                return (
                  <>
                    <Boxuser key={item.id} check={false} infor={item} />
                  </>
                );
              })}
          </div>
        </div>
      </div>
    </>
  );
}

export default Manager;
