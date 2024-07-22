# frozen_string_literal: true

source "https://rubygems.org"

gemspec

group :test do
  gem "html-proofer", "~> 4.4"
end

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]


source "https://rubygems.org"

git_source(:github) { |repo_name| "https://github.com/#{repo_name}" }
gem "jekyll"
gem 'jekyll-feed'
gem 'jekyll-readme-index'
gem 'jemoji'
gem 'webrick'
