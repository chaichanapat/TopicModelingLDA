import TopicModelingLDA as tla

# word = """Exo (Korean: 엑소; stylized in all caps) is a South Korean-Chinese boy band based in Seoul, consisting of nine members: Xiumin, Suho, Lay, Baekhyun, Chen, Chanyeol, D.O., Kai and Sehun. The band was formed by SM Entertainment in 2011 and debuted in 2012. Their music—released and performed in Korean, Mandarin, and Japanese—primarily incorporates pop, hip-hop, and R&B, including elements of electronic dance music genres such as house and trap. The band ranked as one of the top five most influential celebrities on the Forbes Korea Power Celebrity list from 2014 to 2018 and have been labeled "Kings of K-pop" and the "Biggest boyband in the world" by various media outlets.
# The band debuted with twelve members separated into two sub-groups: Exo-K (Suho, Baekhyun, Chanyeol, D.O., Kai, and Sehun) and Exo-M (Xiumin, Lay, Chen, and former members Kris, Luhan, and Tao). Kris, Luhan, and Tao departed the group individually amid legal battles in 2014 and 2015. Exo-K and Exo-M performed music in Korean and Mandarin, respectively, until the release of their third EP Overdose in 2014. Since 2015, Exo have exclusively performed as one group while continuing to release and perform music in multiple languages. Members Chen, Baekhyun, and Xiumin debuted as the sub-unit Exo-CBX in 2016, and members Sehun and Chanyeol began promoting as the sub-unit Exo-SC in 2019. Each member also maintains solo careers in music, film and television.
# Exo's first album XOXO (2013), released alongside breakthrough single "Growl", was a critical and commercial success; it sold over one million copies, which made Exo the first Korean artist to do so in twelve years. Their later works also had strong sales, with their Korean studio albums each selling over one million copies. Exo's sixth album, Don't Mess Up My Tempo (2018), became their highest-charting album on the US Billboard 200, debuting at number 23, and their best selling album in South Korea, where it sold over 1.9 million copies.
# Exo have won numerous awards throughout their career, including five consecutive Album of the Year awards at the Mnet Asian Music Awards and two consecutive Artist of the Year awards at the Melon Music Awards, and have performed over 100 concerts across four headlining tours and multiple joint tours. Outside of music, the band members have endorsement deals with brands such as Nature Republic and Samsung and participate in philanthropic efforts such as Smile For U, an ongoing project by SM Entertainment and UNICEF that began in 2015."""

word = "made made made make make makes make programmers programmers programmers programmers programmer programmings programmings programmings programmings computers computers computer"

numword = 4

if __name__ == '__main__': 
    topics= tla.Predict(word,numword)
    print(topics)


