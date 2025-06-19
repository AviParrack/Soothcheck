# 7/15/24

- Software wants to be useful.

  - Humans decide what software to run based on a judgment it will do something useful.

  - Software that's not useful will not be run.

  - Software that’s not run evaporates away (bitrots) because people stop bothering to run it.

  - In the same way that genes "want" to replicate.

  - If they didn't "want" that then they would not exist.

  - An emergent, inescapable selection pressure and thus inescapable incentive.

- Situated software must individually be small use cases.

  - Because if the use case were viable as an app it would be an app.

  - Apps are large.

  - But it's a mistake to see that each thing individually is small and conclude that therefore the potential for situated software is small.

  - A huge swarm of small things, that in mass is an order of magnitude larger than the use cases that are viable in apps.

- As the market for an experience gets larger, it must increasingly focus on the lowest common denominator: the thing that is at least minimally viable for the largest set of people.

  - As more use cases are absorbed into aggregators, they are less and less able to successfully do things with them.

  - The result is a reduction in the overall number of use cases that are handled in a *great* way.

  - The tyranny of the marginal user.

- Do you think people will make web pages and apps the same way in 5 years in an AI native era?

  - AI changes the friction of production.

  - Think of the edge of apps as a Coasian boundary set by the cost to create a viable piece of software.

  - But now you reduce that cost significantly.

  - How could the boundaries *not* change?

- MidJourney is generating millions of images every day that have never been seen before.

  - The vast majority have an audience of 1.

    - Like going on a hike in the meadow by yourself.

    - A vastly personal experience, just for you.

    - Your own personal museum just for you.

  - If production is so cheap, why bother with distribution?

    - A market of one is fine.

- If software could be fully personalized to you, imagine how much more productive you could be!

  - "Your software". A concept that seems impossible today in our laws of physics.

    - Software is too expensive to make for a market of one.

    - You need to find a collection of users that form a market and design the lowest common denominator for that market.

  - But what if making software was so cheap that it could have a software market of 1?

    - Software whose construction is too cheap to meter.

    - Disposable software.

    - Perfectly bespoke software, perfectly matched to your particular needs in that moment.

- Alvin Toffler had a notion of "prosuming"

  - Producing/consuming at the same time.

  - Somewhat confusingly, this is different than “prosumer” as in “professional-grade consumer”

- What matters most in a given app? The UX or the data?

  - They're normally fundamentally intertwined so you can't tease them apart.

    - A package deal.

  - But imagine that you could separate the UX and the data of an experience for a user. Which one is more strategic?

  - UX is easy to copy.

  - Data is expensive to move across origins and requires trust to put into an origin.

  - The data matters more than the UX.

  - That implies the key thing is doing interesting things for a user with their data.

  - The "compose a little micro app on demand" UX is something that any competent service can do easily.

  - But the hard thing is useful micro-apps that are interesting because they operate on your data.

    - Data you probably don't want to upload willy nilly to some wannabe aggregator.

- The main AI model providers are in an interesting strategic position.

  - Their competitive differentiation is they’re extremely good at producing models.

  - All of them were research orgs first that kind of laid on “sell API access” later… and now find themselves in an unexpected consumer-facing role, too.

  - They started out with a simple demo of their product (the API) in a consumer-facing application.

    - But the model is so good, that even a barebones demo is quite useful!

      - As a user, why go to another tool that wraps the model when the UX is the same?

      - Just go straight to the source, to the default UX.

      - A small advantage, but a consistent one.

    - They start attracting a non-trivial amount of consumer usage.

    - They’ve kind of stumbled backwards into a situation where they could plausibly create the next generation consumer aggregator… or the next generation B2B aggregator (e.g. Salesforce)

    - However their sustainable differentiation is not UX or building an ecosystem, but building a model.

    - And the main labs are acting like they believe there are many orders of magnitude more quality to squeeze out of this model architecture.

    - If the model will get radically better, why waste time on little details and minor side quests that distract from the main job?

  - The result is that these companies have an insanely strong position for traditional aggregator style approaches that any other company would kill for… but don’t necessarily have the time or inclination to execute on *traditional* strategies in that position.

    - For example, as [<u>Simon notes</u>](https://simonwillison.net/2024/Jul/13/give-people-something-to-link-to/), Code Interpreter is insanely powerful, and could be sold today to 100k analysts at a high price… and OpenAI cares about it so little that there’s not even a landing page describing what it is that enthusiastic users can evangelize.

  - The providers all have a consumer UX they host themselves but also an API that allows others to build different experiences powered by the model.

    - If someone could use their model and build a UX that is significantly better, or harnesses the power of an ecosystem to create a system that is the power of the model combined with an ecosystem, then it could break the proto aggregator advantage of the main labs.

    - Other entities can use the same basic model and then layer on a more differentiated / powerful UX, improving upon the model.

    - The importance of an innovative UX or an ecosystem for these labs pales in comparison to the importance of building the next 10x model.

- Anthropic Artifacts is “just” interface sugar, but it’s also transformatively powerful.

  - But sugar that reduces friction can still create a *ton* of value by lowering uncertainty for users to experiment.

  - A UX sugar layer can still be transformative for non-savvy users.

    - It reduces friction and cycle time by an order of magnitude.

    - Take an action, see the result immediately.

    - Encourages experimentation.

  - This week I watched my tech-savvy but not technical friend play around with getting AI to write little toy programs.

    - With OpenAI, I had to copy/paste the generated files into an HTML file for him and refresh the browser to see the output.

      - A somewhat mystifying step to someone who is non technical

    - He engaged with it, getting it to “write a little game with tinkerbell flying around”, but it was frustrating.

    - Then we switched to Anthropic, and seeing the results immediately right there as interactive artifacts led him to 10x his experimentation rate.

- LLMs allow non programmers to get an intuitive handle on the “DNA” of code.

  - That is, the animating logic at its core.

  - They can't write it directly but they can see it being written according to their goals.

    - And so they can develop a vibe they can hack on it, and an intuitive sense of controlling it and guiding it.

  - Before, the only way to do code as a non-programmer was pre-baked examples.

    - There’s not really anything hands-on about creating them, just copy/pasting a bit of code that might as well be a black box.

    - Now LLMs can generate bespoke examples for you on demand, that can then be tweaked with natural language.

  - The non-expert can see the coevolution, feel comfortable with the experimentation, absorb the knowhow; see how the code leads to different outcomes, not by writing it themselves, but having your indefatigable, eager-to-please, knowledgeable friend demonstrate it for you.

  - Will they ever be able to write the code themselves, without the assistance?

    - Who knows.

    - But does it matter if you'll always have that infinitely patient LLM friend there willing and able to help?

    - Similar to the dizzying, terrifying freedom of relying on a tool for thought. "I have superpowers when I use this... what if it went away?"

- A search engine’s quality is determined by different inputs.

  - Those inputs are transformed by an algorithm into the outputs, the Search Engine Results Page (SERP).

  - The inputs for a search engine are:

    - The public crawlable internet

    - The querystream / clickstream of users on the search engine itself.

      - What people search for, and what results users click on.

  - The former is visible to every competitor; anyone can presumably have a similar index (including pagerank calculation) as anyone else.

    - There’s lots of data in the link graph, but it’s data anyone could recreate.

  - But the differentiated inputs are the querystream and clickstream.

    - Those are two extremely high potency signals.

    - The proprietary access to those signals gives a very strong data network effect to the lead search engine.

  - AI has a few similar structural things.

    - The common crawl is (presumably) used by almost every LLM.

    - There’s also proprietary “querystream” of each of the models.

    - When the models are used via the API, most providers contractually agree to not use the queries for training.

    - For the direct consumer application created by the providers, some (like OpenAI) reserve the right to use the querystream to train.

    - Interestingly, if I understand correctly, Anthropic explicitly says they won’t use the querystream to train their models.

- Proto aggregators act like aggregators but are not yet powerful.

  - They’re wanna-be aggregators.

  - "Oh no, we're competing against this proto aggregator, and aggregators are unstoppable!"

    - "They're not an aggregator yet, just a wannabe one. There are lots of moves that can work to head them off. And there are lots of ways that they can be foiled in their path to becoming an aggregator, for example if they don’t have a head and shoulders advantage over competitors.”

- The overriding logic aggregators are forced to follow is to optimize for engagement.

  - If you don't, the competitor that does will suck up all your oxygen and you'll asphyxiate.

  - It is inescapable for aggregators.

  - The internal logic that all employees will bend to is engagement.

  - "We'll optimize for what users want to want, not what they want" sounds nice but is nearly impossible in practice for an aggregator.

  - For a given change, it's hard to tell if it's *good* for users ("who are we to say?") but it's easy to say "this improved 1DA by 2%".

  - The latter is legible and obvious and will get selected for in the organization.

- “We’ll be an aggregator, but a *good* one”

  - You can't be a good aggregator.

  - The problem is the aggregator part!

  - The emergent drive of an aggregator is to absorb all engagement.

  - A gravity well that the ecosystem can’t escape, but also the organization controlling the aggregator can't escape.

  - For the health of the overall ecosystem, a non-aggregator is much better than an aggregator.

  - Unless someone can figure out how to do an open aggregator.

- An open aggregator would be like an aggregator, but good.

  - Aggregators allow zero-friction movement of data within their pocket universe.

    - This unlocks a lot of innovation and value (as long as the aggregator allows 3Ps flexibility to do interesting things).

    - Although the aggregator themselves can, of course, peek whenever they want--you must deeply, fundamentally trust them as a user.

  - Open systems are composed of many different entities.

    - You can’t share data across different entities willy nilly.

  - But if you could change the laws of physics to allow data to move across subcomponents in an open system safely, then it’s possible to get the benefits of an aggregator (low-friction and standardization) with the benefits of open (no one entity in control, no entity that can see all your data, permissionless innovation, ubiquitous self-accelerating ecosystem)

- The same origin paradigm is lots of little silos, so network effects are hard to get going.

  - If you put all of the silos together, it would be unsafe, but have a wild network effect.

    - Remember that network effects go up with the square of size.

  - What if you could make it safe to combine all of the data across origins into one fabric?

- You know what would be cool? Greasemonkey... but safe

  - Greasemonkey had all of this amazing tinkering, creative energy.

  - But it fundamentally could not be made safe. So it had to go away.

  - What if you could make Greasemonkey style tinkering safe?

- Heroes accidentally stunt the growth of people around them

  - When you're eager and capable to do something yourself, it's easy to accidentally leave people behind or disempower them.

  - Because the hero can just do it, and do it faster / better than the other person.

  - But getting good at something takes practice and hands-on experience.

  - If the hero says "Oh I can just do it" or while the other person is huffing and puffing to keep up say "while I'm waiting for you I'll just quickly do this other thing" making the non-hero even more behind.

- Humans learn through peripheral participation in a context.

  - E.g things like apprenticeship, but also things like experimentation.

  - Knowhow / tacit knowledge / intuition.

  - Repeated tedious practice is required for mastery.

    - But why practice when you can just ask the friendly LLM for the answer?

  - If we take away that practice, we could lose an entire generation of talent and knowledge transfer.

  - Before, if you didn’t yet have the knowhow, you could only bother the expert so often, so you had to develop the skill yourself.

    - But now you have LLMs who are always patient and eager to help, and to just heroically give you the right answer.

  - Ethan Mollick has [<u>noted</u>](https://x.com/emollick/status/1777049817585156428) that LLMs will break the implicit apprenticeship model in large organizations.

  - If you've completely stopped reading code, you just copy and paste back and forth between VS Code and Claude and add the text "fix the bug" you're not learning nearly as much any more.

  - This kind of fear is not new; it historically has been brought up for every game-changing technology.

    - Calculators in the past destroyed how math used to be taught.

    - It took us a long time to rebuild how to teach math in a world where everyone had calculators.

    - But with LLMs this process will happen faster… perhaps faster than we can successfully assimilate.

- The more trust a class of thing requires to be viable, the more you'll expect a small number of extremely powerful entities in that class.

  - The asymmetry can be grokked by imagining two options for a calendar scheduling feature.

    - One from a startup you've never met before who you have to give sensitive data to.

    - And the other, a service that *already has* your calendar data.

    - Which one do you pick?

    - The latter, unless the former has an expected value an order of magnitude beyond the former.

  - Over time the one with the edge will grow more and more powerful (thus requiring more and more trust) until only they are left.

- Ben Thompson [<u>points out</u>](https://stratechery.com/2024/aggregators-platforms-and-regulatory-scrutiny-paramount-merges-with-skydance/) that Apple doesn't just do security verification of your app, they say what you can *say* in your app.

  - It’s wild to me that we as a society allow that level of control for the most important devices in people’s lives.

- Narratives are massive dimensionality reducers.

  - They must be!

    - That’s the whole point!

    - Many orders of magnitude of signal reduction into a clean, easy-to-reason-about signal.

  - Pick a frame of interpretation that is compatible with the known information, and then keep the details that "make sense" in that world (i.e. that make the actors in the narrative operate in believable ways following their incentives in that narrative) and throw out the parts that don't.

  - This helps cut through the noise to give us something to work with... but by creating a grotesque, deeply distorted vision of reality.

  - And yet what else could we possibly do?

  - If it were just a cacophony of swirling background noise we'd be frozen in place unable to sense or decide anything, and thus unable to *do* anything with intention.

  - Narratives are our way of grabbing on, of riding this bucking bronco of the real world.

  - The main thing is to never hold too tightly to a narrative, to see it as just one way of seeing the situation, and one that you should be willing to update or change with more disconfirming evidence.

- When viewing from the balcony it's easy to come up with a plan that is simple... and wrong.

  - Narratives are simpler than reality. From a distance, when you don't have direct experience, all you get is narratives.

  - From the floor it's easy to get lost in the shuffle and only be able to think in the immediate short term.

  - You need both perspectives, working together.

- Every organization grows kayfabe up to its carrying capacity.

  - The carrying capacity is set based on viability of business and amount of growth/capital of the main business.

    - Very successful money-making machines can support a large amount of kayfabe.

    - Less successful businesses can support much less kayfabe without dying.

  - The kayfabe grows, emergently, right up to that carrying capacity.

  - And then the organization is left teetering on the edge of criticality, where one wrong step kicks them into chaos and non viability.

  - Kayfabe ratchets.

    - Because the employees who fight it will lose if they give up for a second.

    - As some of the fighters give up, the environment gets less and less hospitable to the people who can fight it.

    - The fight wears out the people who can fight it at an accelerating rate.

    - It's like fighting gravity.

    - Given enough time, gravity will win.

    - Every time.

- Legible doesn't mean "*can* the machine see it" it means "*does* the machine see it".

  - It can be fully in the open but so noisy and chaotic that the vast majority of things aren't seen by the machine.

  - "This doesn't change anything because all of the data was already public anyway" doesn't necessarily track; because it could have been data that could be read and yet wasn't because it wasn't economically viable to do so.

  - LLMs allow sifting through and synthesizing large amounts of data significantly easier than before.

- Any legible medium will over time become increasingly performative.

  - That is, the actions will become increasingly kayfabe.

  - "Human executives are so bad at running orgs. Simply scan all the emails in the organization with an LLM and then decide what to do with their perfect synthesis".

  - But that won’t work for a few reasons.

  - First, it can only operate over things that were written down (or recorded), and many things (e.g. legal things, or kayfabe) won’t be written down.

  - LLMs change what is legible because it changes what is worth it to sift through.

  - If you change the effective legibility (e.g. a thing used to be so hard to sift through that you didn't) with a tool like an LLM, then a previously non-performative medium can become performative.

- Open ecosystems typically coalesce around the simplest possible thing that could possibly kind of work.

  - The more out-there it is, the harder it is to convince the rest of the ecosystem to coordinate around it.

    - Compare with "It's just pubsub, which everything already knows how to talk"

  - It’s less common to see a coherent new ecosystem coalesce around some transformative new tech, and more likely around things like “I just took this off-the-shelf component and baked in a few specific conventions”.

- "I am a good person, and I work here, so this is a good company"

  - People who can see the systemic problems in other systems will miss it in the one they are deeply tied to.

- A general philosophy: get to good enough as quickly as humanly possible and then refine what resonates.

- A kid comes home crying his eyes out because "Toby said he's king of the mountain, but *I'm* king of the mountain!"

  - The adult thinks: “How silly! No one cares if you’re king of the mountain.”

  - But also, how are every day professional power struggles *that* different?

  - Your own power struggles in your context are existential.

  - Ones for others in a different context are petty and inconsequential.

- Help the audience think they came up with the great idea by feeding it to them piece by piece and then letting them feel like a genius.

  - In reality, you staged the insight for them.

  - It felt like an idea maze but it was actually a carefully constructed idea *labyrinth*.

  - A rhetorical cheat, similar to laying out 4 constraints that then the 4 prongs of the solution slot into.

    - People will appreciate the aesthetic “just right” vibe and judge the idea as being better than they otherwise would.

  - Connect nine of the ten dots, the reader feels like a genius for connecting the last dot.

    - And if the full picture is controversial, you have plausible deniability that you intended for that picture to be implied.

- Highly competitive games are expensive, dangerous, and very hard to win.

  - You have to constantly work to be the best of all the rest, with a game that attracts the best talent and the most resources.

    - Blink and you're dead.

  - But if you just need the outcome of the game, but don't care who wins it, you can avoid participating in the whole dangerous and expensive game.

    - You can just benefit from "whoever wins will necessarily produce a cheap and high quality version that I can take for granted".

  - A powerful judo move.

  - The best way to win a hyper competitive game is not to play, but to step adjacent to it to benefit from whoever wins.

  - Related to the "the second meta move to win a gold rush is to design things for a world where there's lots of gold in the economy"

- A philosophy: do things that give you energy that you’re proud of.

- Externalities and indirect effects tend to be diffuse, hard to pin down and measure.

  - But you can get a general vibe of them by asking, "do you feel proud of that?"

  - Imagine being shown a video of that decision in 10 years in front of a thousand people whose opinion you care about.

  - Would you wince?

  - When you're surrounded by fellow optimizers you won’t necessarily be embarrassed of "ignoring the implications", because it’s what everyone else does.

  - But what if you showed it to your grandma and walked her through the decision you made?

- Today users' data flows to an all-powerful aggregator in the sky who you have to trust with your life.

  - The only decision is which aggregator among a handful of mostly similar ones to trust with your life.

  - Why trust a single company with your life?

- When everyone else around you sucks, the bar to clear to be obviously better than everyone is very low.

  - Someone who is better will be a schelling point that many agree is obviously better.

  - That doesn’t have to be that much better in the grand scheme of things if everyone else is obviously worse.

  - The thing that makes something more obviously the right option (and thus the thing that will become the schelling point) is not the absolute amount of quality, it’s how *obvious* it is that it’s different.

    - This is related but distinct!

    - Being higher absolute quality will tend to make it more obvious that it’s higher quality, but not always.

- The entity that has all the power is allowed to be clueless.

  - To summarize (my understanding of) the main thrust of *[<u>Jane Austen, Game Theorist</u>](https://press.princeton.edu/books/hardcover/9780691155760/jane-austen-game-theorist):*

    - The men are clueless, because they can be.

    - The women have no recourse but to do clever strategies to get ahead.

  - Why is the tech industry not particularly self-reflective about the implications of its actions?

    - Because the tech industry has all of the money and power.

    - The tech industry is clueless because it can be.

- Have 2 hours to do a 5 minute task?

  - Impossible effort of will.

  - 5 minutes to do a 6 minute task?

  - Let's go!

  - When you’re time constrained, there’s no way to delay the start of it, you just have to go.

  - Having more time can make you less productive, “I can delay that by a few more minutes, just *one* more YouTube video…”

  - One good way to keep the time urgency up: spend a lot of time in your creative / flow state, so you have only a small amount of time to do the various annoying simple chores.

  - The momentum from the creative work, and the small amount of time available for the annoying tasks, helps you get them done lickety split.

- Using LLMs effectively today requires skill to know how to drive them effectively.

  - But over time to become mass market it has to be a forgiving thing that anyone can do.

  - How can you make it so more people can wield LLMs effectively without being wizards?

- When you copy/paste data from one system to another as a user, you're doing a manual, implied integrity check.

  - "Do I trust this app with this data?"

  - Natural and in context (but also imprecise and prone to errors of judgment and a little scary).

- The Bitter Lesson: Which wins, the one who throws algorithm effort at the problem, or more data?

  - The bitter lesson is that more data always wins.

  - An ecosystem approach to quality is effectively throwing more improvement at the system than any algorithm designer could do linearly.

- The Ouija board effect: minor amplitude but consistently aligned phenomena lead to significant movement of the collective, as if by magic.

  - "Where is the singular cause of this obvious macro behavior? There isn't one, therefore it must be magic!"

  - The trick is that the bits of variance are small and lost in the noise.

  - But because the variance is aligned across multiple entities, they stand out from the background noise strongly.

  - Not strength of the variance, but the alignment.

  - In particular in Ouija, the indicator starts off moving randomly, but the people moving it are interpreting it in a continuous feedback loop. As it starts looking like it’s moving to a letter, if it makes sense, then it starts going there faster and faster as more of the people with hands on the indicator become aware of the implied hypothesis of the next letter and think it’s plausible.

  - Other people will think magic is impossible and under-utilize this strategy.

  - But if you see that magic is real, you can deploy it very intentionally.

- What we all say enough becomes true.

  - It's not obvious that this should be correct.

  - But we change our opinions on what to expect based on what we hear.

  - The things that happen coevolve their possibility space with what people believe.

  - If enough people believe it may happen (and take consistent, if minor, actions to prepare for it possibly happening), that makes it more likely to happen.

  - Small amounts of variance individually, but aligned with that thing coming true, which makes it significantly more likely to come true.