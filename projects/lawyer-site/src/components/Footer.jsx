const Footer = () => {
  const currentYear = new Date().getFullYear()

  const footerLinks = {
    services: [
      { name: 'Corporate Law', href: '#services' },
      { name: 'Family Law', href: '#services' },
      { name: 'Personal Injury', href: '#services' },
      { name: 'Criminal Defense', href: '#services' },
      { name: 'Estate Planning', href: '#services' },
      { name: 'Real Estate Law', href: '#services' },
    ],
    company: [
      { name: 'About Us', href: '#about' },
      { name: 'Our Team', href: '#about' },
      { name: 'Careers', href: '#' },
      { name: 'News & Insights', href: '#' },
      { name: 'Contact', href: '#contact' },
    ],
    resources: [
      { name: 'Client Portal', href: '#' },
      { name: 'Legal Resources', href: '#' },
      { name: 'FAQs', href: '#' },
      { name: 'Blog', href: '#' },
    ],
  }

  return (
    <footer className="bg-navy-950 border-t border-navy-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-12">
          <div className="lg:col-span-2">
            <a href="#home" className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 bg-gold-500 rounded-sm flex items-center justify-center">
                <span className="text-navy-950 font-serif font-bold text-xl">S</span>
              </div>
              <div>
                <span className="text-white font-serif text-xl font-semibold tracking-wide">Sterling</span>
                <span className="text-gold-500 font-serif text-xl font-light"> & Associates</span>
              </div>
            </a>
            <p className="text-navy-400 leading-relaxed max-w-sm">
              Dedicated to providing exceptional legal representation with integrity, professionalism,
              and a commitment to achieving the best outcomes for our clients.
            </p>

            <div className="flex space-x-4 mt-6">
              {[
                { name: 'LinkedIn', icon: 'M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z' },
                { name: 'Twitter', icon: 'M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z' },
                { name: 'Facebook', icon: 'M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385h-3.047v-3.47h3.047v-2.642c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953h-1.514c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385c5.737-.9 10.125-5.864 10.125-11.854z' },
              ].map((social) => (
                <a
                  key={social.name}
                  href="#"
                  className="w-10 h-10 bg-navy-800 hover:bg-gold-500 rounded-full flex items-center justify-center text-navy-400 hover:text-navy-950 transition-all duration-200"
                  aria-label={social.name}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d={social.icon} />
                  </svg>
                </a>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Practice Areas</h4>
            <ul className="space-y-3">
              {footerLinks.services.map((link) => (
                <li key={link.name}>
                  <a href={link.href} className="text-navy-400 hover:text-gold-500 transition-colors text-sm">
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Company</h4>
            <ul className="space-y-3">
              {footerLinks.company.map((link) => (
                <li key={link.name}>
                  <a href={link.href} className="text-navy-400 hover:text-gold-500 transition-colors text-sm">
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-3">
              {footerLinks.resources.map((link) => (
                <li key={link.name}>
                  <a href={link.href} className="text-navy-400 hover:text-gold-500 transition-colors text-sm">
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-navy-800">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-navy-500 text-sm">
              © {currentYear} Sterling & Associates. All rights reserved.
            </div>
            <div className="flex flex-wrap justify-center gap-6 text-sm">
              <a href="#" className="text-navy-500 hover:text-gold-500 transition-colors">
                Privacy Policy
              </a>
              <a href="#" className="text-navy-500 hover:text-gold-500 transition-colors">
                Terms of Service
              </a>
              <a href="#" className="text-navy-500 hover:text-gold-500 transition-colors">
                Disclaimer
              </a>
              <a href="#" className="text-navy-500 hover:text-gold-500 transition-colors">
                Accessibility
              </a>
            </div>
          </div>
          <div className="mt-6 text-center text-navy-600 text-xs">
            <p>
              Attorney Advertising. Prior results do not guarantee a similar outcome.
              This website is for informational purposes only and does not constitute legal advice.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
