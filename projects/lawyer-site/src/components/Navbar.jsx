import { useState } from 'react'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  const navLinks = [
    { name: 'Home', href: '#home' },
    { name: 'Services', href: '#services' },
    { name: 'About', href: '#about' },
    { name: 'Contact', href: '#contact' },
  ]

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-navy-950/95 backdrop-blur-sm border-b border-navy-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center">
            <a href="#home" className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gold-500 rounded-sm flex items-center justify-center">
                <span className="text-navy-950 font-serif font-bold text-xl">S</span>
              </div>
              <div className="hidden sm:block">
                <span className="text-white font-serif text-xl font-semibold tracking-wide">Sterling</span>
                <span className="text-gold-500 font-serif text-xl font-light"> & Associates</span>
              </div>
            </a>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <a
                key={link.name}
                href={link.href}
                className="text-navy-200 hover:text-gold-500 transition-colors duration-200 text-sm font-medium tracking-wide uppercase"
              >
                {link.name}
              </a>
            ))}
            <a
              href="#contact"
              className="bg-gold-600 hover:bg-gold-500 text-navy-950 px-6 py-2.5 text-sm font-semibold tracking-wide uppercase transition-colors duration-200"
            >
              Free Consultation
            </a>
          </div>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-white p-2"
            aria-label="Toggle menu"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
        </div>

        {isOpen && (
          <div className="md:hidden pb-4">
            <div className="flex flex-col space-y-3">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className="text-navy-200 hover:text-gold-500 transition-colors duration-200 text-sm font-medium tracking-wide uppercase py-2"
                >
                  {link.name}
                </a>
              ))}
              <a
                href="#contact"
                onClick={() => setIsOpen(false)}
                className="bg-gold-600 hover:bg-gold-500 text-navy-950 px-6 py-2.5 text-sm font-semibold tracking-wide uppercase transition-colors duration-200 text-center mt-2"
              >
                Free Consultation
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navbar
