import { Route,BrowserRouter ,Routes} from "react-router-dom"
import Home from './pages/Home'
import Navbar from "./components/Navbar"
import Login from './pages/login'
import About from './pages/about'
import PYBMarketPlace from './pages/Userinfo'
const App = () => {
  return (
  <main className="bg-slate-300/20">
    <BrowserRouter>
    <Navbar/>
    <Routes>
    <Route path="/"  element={<Home/>}/>
    <Route path="/about" element={<About/>}/>
    <Route path="/login" element={<Login />} />
    <Route path="/userinfo" element={<PYBMarketPlace />} />
    </Routes>
    </BrowserRouter>
  </main>
  )
}

export default App
