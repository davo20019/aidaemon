const services = [
  {
    title: 'Corporate Law',
    description: 'Comprehensive legal solutions for businesses of all sizes. From formation and contracts to mergers and compliance, we protect your business interests.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
      </svg>
    ),
    features: ['Business Formation', 'Contract Drafting', 'Mergers & Acquisitions', 'Regulatory Compliance'],
  },
  {
    title: 'Family Law',
    description: 'Compassionate guidance through life\'s most challenging moments. We handle divorce, custody, adoption, and all family legal matters with care.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
      </svg>
    ),
    features: ['Divorce & Separation', 'Child Custody', 'Adoption Services', 'Prenuptial Agreements'],
  },
  {
    title: 'Personal Injury',
    description: 'Fighting for the compensation you deserve. Our aggressive representation ensures maximum recovery for accident victims and their families.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    features: ['Auto Accidents', 'Medical Malpractice', 'Workplace Injuries', 'Wrongful Death'],
  },
  {
    title: 'Criminal Defense',
    description: 'Protecting your rights and freedom when it matters most. Our experienced defense attorneys provide aggressive representation in all criminal matters.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
      </svg>
    ),
    features: ['DUI Defense', 'White Collar Crime', 'Drug Offenses', 'Federal Cases'],
  },
  {
    title: 'Estate Planning',
    description: 'Secure your legacy and protect your loved ones. We create comprehensive estate plans tailored to your unique circumstances and goals.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    features: ['Wills & Trusts', 'Probate Administration', 'Asset Protection', 'Healthcare Directives'],
  },
  {
    title: 'Real Estate Law',
    description: 'Expert guidance for all property transactions. From residential purchases to commercial developments, we ensure smooth closings and clear titles.',
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
    features: ['Residential Closings', 'Commercial Leasing', 'Title Disputes', 'Zoning Issues'],
  },
]

const Services = () => {
  return (
    <section id="services" className="py-24 bg-navy-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <span className="text-gold-600 text-sm font-semibold tracking-widest uppercase">Our Practice Areas</span>
          <h2 className="mt-4 text-3xl sm:text-4xl lg:text-5xl font-serif font-bold text-navy-900">
            Comprehensive Legal Services
          </h2>
          <p className="mt-4 text-navy-600 text-lg max-w-2xl mx-auto">
            Our experienced attorneys provide expert representation across a wide range of practice
            areas, ensuring you receive the specialized attention your case deserves.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <div
              key={index}
              className="group bg-white rounded-lg p-8 shadow-sm hover:shadow-xl transition-all duration-300 border border-navy-100 hover:border-gold-500/50"
            >
              <div className="w-16 h-16 bg-navy-900 rounded-lg flex items-center justify-center text-gold-500 group-hover:bg-gold-500 group-hover:text-navy-900 transition-colors duration-300">
                {service.icon}
              </div>

              <h3 className="mt-6 text-xl font-serif font-semibold text-navy-900 group-hover:text-gold-600 transition-colors duration-200">
                {service.title}
              </h3>

              <p className="mt-3 text-navy-600 leading-relaxed">
                {service.description}
              </p>

              <ul className="mt-6 space-y-2">
                {service.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="flex items-center text-sm text-navy-500">
                    <svg className="w-4 h-4 text-gold-500 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              <a
                href="#contact"
                className="inline-flex items-center mt-6 text-gold-600 font-medium text-sm hover:text-gold-700 transition-colors group-hover:underline"
              >
                Learn More
                <svg className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </a>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <p className="text-navy-600 mb-6">
            Don't see what you're looking for? We handle many additional practice areas.
          </p>
          <a
            href="#contact"
            className="inline-flex items-center justify-center bg-navy-900 hover:bg-navy-800 text-white px-8 py-4 text-sm font-semibold tracking-wide uppercase transition-all duration-200"
          >
            Contact Us to Discuss Your Case
          </a>
        </div>
      </div>
    </section>
  )
}

export default Services
