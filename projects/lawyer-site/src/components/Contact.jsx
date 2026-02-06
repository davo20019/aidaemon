import { useState } from 'react'

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    service: '',
    message: '',
  })
  const [submitted, setSubmitted] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    setSubmitted(true)
  }

  return (
    <section id="contact" className="py-24 bg-navy-950 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-96 h-96 bg-gold-500/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-gold-500/5 rounded-full blur-3xl" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="text-center mb-16">
          <span className="text-gold-500 text-sm font-semibold tracking-widest uppercase">Get in Touch</span>
          <h2 className="mt-4 text-3xl sm:text-4xl lg:text-5xl font-serif font-bold text-white">
            Schedule Your Free Consultation
          </h2>
          <p className="mt-4 text-navy-300 text-lg max-w-2xl mx-auto">
            Take the first step toward resolving your legal matter. Our team is ready to listen,
            advise, and advocate for your rights.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-12">
          <div className="lg:col-span-2">
            {submitted ? (
              <div className="bg-navy-900 border border-navy-700 rounded-lg p-12 text-center">
                <div className="w-20 h-20 mx-auto bg-gold-500/20 rounded-full flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-gold-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="text-2xl font-serif font-semibold text-white mb-4">Thank You!</h3>
                <p className="text-navy-300">
                  We've received your consultation request. A member of our team will contact you
                  within 24 hours to discuss your case.
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="bg-navy-900 border border-navy-700 rounded-lg p-8">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-navy-200 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      required
                      className="w-full bg-navy-800 border border-navy-600 rounded-lg px-4 py-3 text-white placeholder-navy-400 focus:outline-none focus:border-gold-500 focus:ring-1 focus:ring-gold-500 transition-colors"
                      placeholder="John Smith"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-navy-200 mb-2">
                      Email Address *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="w-full bg-navy-800 border border-navy-600 rounded-lg px-4 py-3 text-white placeholder-navy-400 focus:outline-none focus:border-gold-500 focus:ring-1 focus:ring-gold-500 transition-colors"
                      placeholder="john@example.com"
                    />
                  </div>
                  <div>
                    <label htmlFor="phone" className="block text-sm font-medium text-navy-200 mb-2">
                      Phone Number
                    </label>
                    <input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      className="w-full bg-navy-800 border border-navy-600 rounded-lg px-4 py-3 text-white placeholder-navy-400 focus:outline-none focus:border-gold-500 focus:ring-1 focus:ring-gold-500 transition-colors"
                      placeholder="(555) 123-4567"
                    />
                  </div>
                  <div>
                    <label htmlFor="service" className="block text-sm font-medium text-navy-200 mb-2">
                      Type of Legal Matter *
                    </label>
                    <select
                      id="service"
                      name="service"
                      value={formData.service}
                      onChange={handleChange}
                      required
                      className="w-full bg-navy-800 border border-navy-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-gold-500 focus:ring-1 focus:ring-gold-500 transition-colors"
                    >
                      <option value="">Select a practice area</option>
                      <option value="corporate">Corporate Law</option>
                      <option value="family">Family Law</option>
                      <option value="injury">Personal Injury</option>
                      <option value="criminal">Criminal Defense</option>
                      <option value="estate">Estate Planning</option>
                      <option value="realestate">Real Estate Law</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>
                <div className="mt-6">
                  <label htmlFor="message" className="block text-sm font-medium text-navy-200 mb-2">
                    Tell Us About Your Case *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    required
                    rows={5}
                    className="w-full bg-navy-800 border border-navy-600 rounded-lg px-4 py-3 text-white placeholder-navy-400 focus:outline-none focus:border-gold-500 focus:ring-1 focus:ring-gold-500 transition-colors resize-none"
                    placeholder="Please provide a brief description of your legal matter..."
                  />
                </div>

                <div className="mt-6 flex items-start">
                  <input
                    type="checkbox"
                    id="privacy"
                    required
                    className="mt-1 h-4 w-4 text-gold-500 border-navy-600 rounded focus:ring-gold-500 bg-navy-800"
                  />
                  <label htmlFor="privacy" className="ml-3 text-sm text-navy-400">
                    I agree to the privacy policy and consent to being contacted regarding my inquiry.
                    All information shared is confidential and protected by attorney-client privilege.
                  </label>
                </div>

                <button
                  type="submit"
                  className="mt-8 w-full bg-gold-600 hover:bg-gold-500 text-navy-950 px-8 py-4 text-sm font-semibold tracking-wide uppercase transition-all duration-200 rounded-lg hover:shadow-lg hover:shadow-gold-500/20"
                >
                  Request Free Consultation
                </button>
              </form>
            )}
          </div>

          <div className="space-y-8">
            <div className="bg-navy-900 border border-navy-700 rounded-lg p-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-gold-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-gold-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Main Office</h4>
                  <p className="text-navy-300 text-sm">
                    123 Legal Plaza, Suite 500<br />
                    New York, NY 10001
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-navy-900 border border-navy-700 rounded-lg p-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-gold-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-gold-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Call Us</h4>
                  <p className="text-navy-300 text-sm">
                    <a href="tel:+1-800-555-0123" className="hover:text-gold-500 transition-colors">
                      (800) 555-0123
                    </a>
                  </p>
                  <p className="text-navy-400 text-xs mt-1">Available 24/7 for emergencies</p>
                </div>
              </div>
            </div>

            <div className="bg-navy-900 border border-navy-700 rounded-lg p-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-gold-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-gold-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Email Us</h4>
                  <p className="text-navy-300 text-sm">
                    <a href="mailto:contact@sterlinglaw.com" className="hover:text-gold-500 transition-colors">
                      contact@sterlinglaw.com
                    </a>
                  </p>
                  <p className="text-navy-400 text-xs mt-1">Response within 24 hours</p>
                </div>
              </div>
            </div>

            <div className="bg-navy-900 border border-navy-700 rounded-lg p-6">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-gold-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <svg className="w-6 h-6 text-gold-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-1">Office Hours</h4>
                  <p className="text-navy-300 text-sm">
                    Monday - Friday: 8:30 AM - 6:00 PM<br />
                    Saturday: 9:00 AM - 1:00 PM<br />
                    Sunday: Closed
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Contact
