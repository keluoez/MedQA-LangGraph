import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import zh from "./locales/zh.json";
import en from "./locales/en.json";

const saved = (typeof localStorage !== "undefined" && localStorage.getItem("lang")) as
  | "zh"
  | "en"
  | null;

i18n.use(initReactI18next).init({
  resources: {
    zh: { translation: zh },
    en: { translation: en },
  },
  lng: saved ?? "zh",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export default i18n;
