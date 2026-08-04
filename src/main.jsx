import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./App.css";
import { installErrorReporting } from "./report.js";

installErrorReporting();

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
