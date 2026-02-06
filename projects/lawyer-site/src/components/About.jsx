const team = [
  {
    name: 'Victoria Sterling',
    role: 'Managing Partner',
    specialty: 'Corporate Law',
    image: 'VS',
  },
  {
    name: 'Michael Chen',
    role: 'Senior Partner',
    specialty: 'Criminal Defense',
    image: 'MC',
  },
  {
    name: 'Sarah Williams',
    role: 'Partner',
    specialty: 'Family Law',
    image: 'SW',
  },
  {
    name: 'James Morrison',
    role: 'Partner',
    specialty: 'Personal Injury',
    image: 'JM',
  },
]

const About = () => {
  return (
    <section id="about" className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <span className="text-gold-600 text-sm font-semibold tracking-widest uppercase">About Our Firm</span>
            <h2 className="mt-4 text-3xl sm:text-4xl lg:text-5xl font-serif font-bold text-navy-900">
              A Legacy of Legal Excellence
            </h2>
            <p className="mt-6 text-navy-600 text-lg leading-relaxed">
              Founded in 1989, Sterling & Associates has grown from a small practice to one of the
              region's most respected law firms. Our commitment to client success, ethical practice,
              and legal innovation has earned us recognition as leaders in our field.
            </p>
            <p className="mt-4 text-navy-600 text-lg leading-relaxed">
              We believe that everyone deserves access to exceptional legal representation. Our
              diverse team of attorneys brings together decades of combined experience, ensuring
              that no matter how complex your legal challenge, we have the expertise to help.
            </p>

            <div className="mt-8 grid grid-cols-2 gap-6">
              <div className="border-l-4 border-gold-500 pl-4">
                <div className="text-3xl font-serif font-bold text-navy-900">50+</div>
                <div className="text-navy-600 text-sm mt-1">Legal Professionals</div>
              </div>
              <div className="border-l-4 border-gold-500 pl-4">
                <div className="text-3xl font-serif font-bold text-navy-900">10,000+</div>
                <div className="text-navy-600 text-sm mt-1">Cases Handled</div>
              </div>
              <div className="border-l-4 border-gold-500 pl-4">
                <div className="text-3xl font-serif font-bold text-navy-900">6</div>
                <div className="text-navy-600 text-sm mt-1">Office Locations</div>
              </div>
              <div className="border-l-4 border-gold-500 pl-4">
                <div className="text-3xl font-serif font-bold text-navy-900">35+</div>
                <div className="text-navy-600 text-sm mt-1">Years of Service</div>
              </div>
            </div>

            <div className="mt-10 flex flex-wrap gap-4">
              <div className="flex items-center space-x-2 bg-navy-50 px-4 py-2 rounded-full">
                <svg className="w-5 h-5 text-gold-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="text-navy-700 text-sm font-medium">AV Preeminent Rated</span>
              </div>
              <div className="flex items-center space-x-2 bg-navy-50 px-4 py-2 rounded-full">
                <svg className="w-5 h-5 text-gold-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="text-navy-700 text-sm font-medium">Super Lawyers</span>
              </div>
              <div className="flex items-center space-x-2 bg-navy-50 px-4 py-2 rounded-full">
                <svg className="w-5 h-5 text-gold-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="text-navy-700 text-sm font-medium">Best Law Firms</span>
              </div>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-4 bg-gradient-to-r from-gold-500/20 to-navy-900/20 rounded-lg transform -rotate-2" />
            <div className="relative bg-navy-900 rounded-lg p-8">
              <h3 className="text-white font-serif text-2xl font-semibold mb-8">Meet Our Partners</h3>
              <div className="grid grid-cols-2 gap-6">
                {team.map((member, index) => (
                  <div key={index} className="text-center">
                    <div className="w-20 h-20 mx-auto bg-gradient-to-br from-gold-500 to-gold-600 rounded-full flex items-center justify-center mb-4">
                      <span className="text-navy-900 font-serif font-bold text-xl">{member.image}</span>
                    </div>
                    <h4 className="text-white font-semibold">{member.name}</h4>
                    <p className="text-gold-500 text-sm">{member.role}</p>
                    <p className="text-navy-400 text-xs mt-1">{member.specialty}</p>
                  </div>
                ))}
              </div>

              <div className="mt-8 pt-6 border-t border-navy-700">
                <p className="text-navy-300 text-sm text-center italic">
                  "Our mission is to provide exceptional legal representation while maintaining the
                  highest standards of integrity and professionalism."
                </p>
                <p className="text-gold-500 text-sm text-center mt-2">— Victoria Sterling, Managing Partner</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-24">
          <div className="text-center mb-12">
            <h3 className="text-2xl font-serif font-semibold text-navy-900">Why Choose Sterling & Associates?</h3>
          </div>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ),
                title: 'Responsive Service',
                description: 'We return all calls within 24 hours and keep you informed every step of the way.',
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                ),
                title: 'Proven Results',
                description: 'Our track record speaks for itself with millions recovered for our clients.',
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                ),
                title: 'Personal Attention',
                description: 'You work directly with experienced attorneys, not assistants or paralegals.',
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ),
                title: 'Transparent Fees',
                description: 'Clear, upfront pricing with no hidden costs. Many cases on contingency.',
              },
            ].map((item, index) => (
              <div key={index} className="text-center">
                <div className="w-16 h-16 mx-auto bg-navy-100 rounded-full flex items-center justify-center text-gold-600 mb-4">
                  {item.icon}
                </div>
                <h4 className="text-navy-900 font-semibold mb-2">{item.title}</h4>
                <p className="text-navy-600 text-sm">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export default About
