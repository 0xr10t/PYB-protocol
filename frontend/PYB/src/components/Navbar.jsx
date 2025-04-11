import { NavLink } from "react-router-dom"
import logo from '../assets/logo.png'

const Navbar = () => {
  return (
    <header className='header'>
      <nav className='flex items-center text-lg gap-7 font-medium'>
        <NavLink to='/' className="mr-4">
          <img src={logo} alt="PYB Logo" className="h-16 w-auto" />
        </NavLink>
        <NavLink to='/about' className={({ isActive }) => isActive ? "text-blue-600" : "text-black" }>
          About
        </NavLink>
        <NavLink to='/login' className={({ isActive }) => isActive ? "text-blue-600" : "text-black"}>
          Get Started
        </NavLink>
      </nav>
    </header>
  )
}

export default Navbar