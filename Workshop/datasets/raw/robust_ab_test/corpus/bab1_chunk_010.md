# 9/30/24

- One of the best use cases for LLMs: they can compile English to code.

  - The implications of this are significant.

  - English is at a much higher pace layer than traditional code, which allows moving more quickly and experimenting with much higher leverage.

  - It used to be that taking an English language description and converting it into code that a computer could actually run was an expensive, tedious, extremely high-skill task of generalization and translation.

  - But now for software in the small it takes just a few seconds.

  - As Simon Willison [<u>notes</u>](https://x.com/simonw/status/1837179899829985437), the people best situated for this world are people who both know how to code and also have the knowhow of using LLMs.

  - A couple of weeks ago I [<u>created a little artifact</u>](https://komoroske.com/skyjo) to help with strategy when playing the card game Skyjo.

    - Writing that little artifact was both very much like programming and also very much not.

    - I knew how *I* would have tackled such a project, so I fed the LLM iterative steps, starting with the core of the card counting logic and then adding up features and tweaking them.

    - But instead of me doing the work, I was just using my knowhow to guide it, and it took a couple of minutes to get precisely the output I wanted instead of hours.

    - That's the real knowhow of software engineering: What is the simplest possible slice of a thing that will work quickly, that can then be incrementally improved / extended into a production-caliber, full featured thing?

- Normal LLM models are like the brain’s System 1.

  - That is, highly parallel, vibes matching from past experience.

  - OpenAI’s o1 (codenamed Strawberry) is different: like the brain’s System 2.

  - Heavy, expensive machinery for general purpose problem solving.

  - Some problems are one-ply problems.

    - These are problems that the right expert could give a gut answer on and be right.

    - Normal LLMs are great at these.

  - Some problems are inherently multi-ply problems.

    - Even experts in the field would have to sit down to think it through

    - This is where something like o1 is useful.

  - In normal LLMs, when generating output, it’s YOLOing tokens.

    - If its gut answer to an early token is wrong, the rest of the answer will also now be trash.

    - Vs o1 can recognize it made an error and revise it.

  - o1 style models are less about outcome supervision vs process supervision.

    - Having an exceptionally well tuned process can be very powerful.

    - In high school, one of my favorite teachers was my AP Physics teacher, Dr. Patel.

    - On the very first day he gave all of us the answer key of the final answer to every question in the book.

    - He’d grade our homework on how rigorously we followed the process, not the right answer (which we already knew).

      - Every single formula, input, intermediate step had to be shown in order, cleanly written.

      - He’d grade *extraordinarily* harshly.

    - At the beginning it drove me crazy.

    - But as time went on and I got good at it it felt like flying; no matter how complicated the problem I was confident I could break it down into smaller pieces until it yielded to the process.

    - o1’s training is like giving the model its own Dr. Patel.

  - Whereas working with other LLMs is a little kayak, o1 is a massive ship.

    - Use it when you want to pull out the big guns.

    - Or when it's worth it to write a mini spec of what to do and come back later when it's done.

  - It used to be that to get great results out of LLMs required a lot of prompt-fu expertise.

    - This system can give very good results no matter how good the prompt is, but is hard for people with prompt-fu to steer.

- o1 is not like other LLMs; it’s a different type of thing.

  - It's more akin to comparing whole systems/scaffolding (e.g. Github Copilot Workspaces) *around* a model than comparing base models.

  - It just so happens to implement what other people have implemented as a system around a model, into the model itself.

  - o1 is a monolith, but a self-improving monolith. Produced by a generative process, not by humans.

  - That gives it some benefits: the ability to train it to do better (possibly without limit, if you want to invest tons of CPU into it, like Alpha Go).

  - But it also means that it's "integrated" and hard to tweak and direct and configure.

    - One of the dials you can't get into in o1: how long to think about this task?

  - A more modular system discovered by a swarming ecosystem still could do better.

- No one thinks that calculators have an inner mind.

  - But it's easy to believe an LLM does!

  - An LLM is like a calculator for words… but it can have a conversation with you.

  - It’s hard to avoid falling for the illusion that it’s a person just like you.

- Just because LLMs can talk doesn't mean your app should have them talk.

  - A lot of UX patterns today using LLMs imply working with a human.

  - When there's an implied human you have to reason through "what is its personality? its goals? its abilities? What is it thinking? Will it judge me for this question?"

  - But it’s also possible to have a system that can understand your plain language thoughts and act on them... but doesn't have a personality.

  - Spellcheck and Github Copilot are two examples.

  - It doesn't feel like you're working with a genie, it's just instantly offering completions you can accept or not.

  - The question isn't "what is this agent thinking", but just "is this autocompletion useful or not."

  - You can still use LLMs and their judgment to produce better suggestions; using the LLM in the back office, not the front office.

- The only way to learn how to use LLMs is to get hands-on experience with them.

  - The open-endedness is intimidating!

  - But the more you use LLMs, the more of a feel for them you get, the better you’re able to drive them and use them effectively.

- Your data, enchanted.

  - Not about apps, but about your data.

  - Instead of your data sitting around passively, it becomes ever more useful to you in that moment, sprouting more functionality.

  - The more you collect and tend it, the more magic it gets.

  - Imagine collecting your data to be the fertile soil, where spontaneous apps sprout automatically, creating hyper bespoke seedlings of possibility.

  - All just for you.

- I built the [<u>system</u>](https://github.com/jkomoros/card-web) behind [<u>https://thecompendium.cards</u>](https://thecompendium.cards) over the course of many years.

  - I coevolved it with my own usage, adding features that fit my own bespoke needs like a glove.

    - I have thousands of private working notes that I develop each week; it’s unthinkable for me to imagine giving it up.

  - Imagine how amazing it could be if everyone could have their own personal software like my Compendium is for me?

  - Imagine if they could sprout up automatically, with only a bit of gardening?

- A tool to build software sounds like it’s for developers

  - "Oh, I don't have that problem... I don't even know how to write software!"

  - And yet everyone could use a system that generates personal software for you automatically.

- “I made this bit of software in 3 hours using an LLM, I should turn it into a business!”

  - “No, you’re assuming software is expensive. It’s now cheap! That changes everything!”

  - When LLMs can help build software quickly, execution velocity on building a bit of software becomes less of a differentiator.

  - We’ll see less “one size fits all gleaming efficient software“ more “messy but effective hyper niche software”

- Most software for specific needs is very simple.

  - The complexity of software goes up super linearly as it needs to be generalized.

  - Handling edge cases is the hard part of software and the more you generalize a bit of software the more edge cases that have to be handled.

  - Before we had compilers from english to code, software was always expensive, so to have the investment in a bit of software make sense it had to have a broader audience, which required generalization.

    - And if you were going to generalize, you might as well polish the experience to a nice shine.

  - People haven’t internalized how amazing it is to have compilers for English to software in the small.

  - Small niche software is very low complexity, and LLMs can do a very good job with it.

- Why are applications the current “size”?

  - That is, what determines whether we have lots of little, specific apps or a small number of large, general purpose ones?

  - Probably a lot of factors, but one that I think is important is what I’d call the Coasian theory of the app.

  - That is, the app size is determined partially by the coordination cost of integrating your app with another party’s system.

  - Code is unforgiving; when two systems need to interact it’s like finely machined gears that have to mesh perfectly with the gears around them.

  - If you made both of the gears, it’s easy to make sure they enmesh, and also easier to modify both at the same time.

  - But if your gear has to enmesh with someone else’s gear (or maybe a lot of different someone’s) then it’s more of a pain and you might decide to bring the other side in house too.

  - Part of this cost is “how expensive is it to write code to enmesh well with the APIs of external systems.”

  - But now we can compile english to code.

  - So in a post-app world that is LLM-native, you might expect to see a larger number of smaller apps.

- LLMs take away some of the costs of producing output.

  - If you liked the actual experience of production (as an artist or programmer) then it's scary, it will take the parts you liked.

  - But if you saw the point of being an artist or programmer was the *output*, and the process of production was just a means to an end, you might not care.

  - Before LLMs you had to love the production to stick with it in order to gain skills and produce output you were proud of, but now focusing on the production could prevent you from a new style of faster exploration.

  - In such a world, the taste and judgment on what constitutes good output is more important than the ability to do the production.

- Greasemonkey was wonderful back in the day.

  - The amount of use agency, the ability to tinker, to not have to put up with a one-size-fits-all app experience, but to be able to customize it to work for you.

  - It had amazing generative potential of an ecosystem that was self-catalyzing.

  - But Greasemonkey ultimately had a low ceiling due to its security model.

    - The laws of physics of a system, its security model, sets both the speed of distribution and also the ceiling of max penetration.

  - Imagine a Greasemonkey for the post-app era that didn’t have such a low ceiling.

- Don't bother making a decentralized clone of a thing that already exists.

  - That's a nearly impossible slog, with little payoff.

  - The same problems of distribution, centralization, retention... but with no user value and even harder because of more complex architecture.

  - A decentralized, emergent system will have to enable *new* possibilities.

- A nice piece from my friend Matt Webb: [<u>Sometimes the product innovation is the distribution</u>](https://interconnected.org/home/2024/09/27/distribution?utm_source=genmon&utm_medium=email&utm_campaign=interconnected-sometimes-the-product-innovation).

- Spontaneous software is clearly a high potential use for LLMs.

  - There are lots of amazing systems that have sprouted up for making spontaneous software.

  - But all of those existing systems so far are about creating a little app that could exist today.

  - But these little app like experiences aren't actually game changing yet, because they still need to be distributed and work with real data.

  - So these tools currently all produce little toys and proofs of concept, but not things that are plugged into real systems to do real work.

  - LLMs can make software in the small quickly. You need a way of distributing that can safely work on your data with low friction to unlock the potential of LLMs.

- Imagine there’s a new system that opens up new possibilities unlike what are in other systems by massively reducing the friction to conjure up new experiences.

  - How would you know it actually was different?

  - One way is demos of use cases that would be unthinkable in other systems, where use cases are more expensive to make.

  - What is the most frivolous use case you can imagine?

  - Imagine looking at your personal calendar data and whispering in its ear “make it into [<u>a breakout game</u>](https://en.wikipedia.org/wiki/Breakout_(video_game)),” and it does.

  - Only an open-ended generative system that makes bespoke software easy would ever be able to do that.

  - No one would bother writing up such frivolous software on their own.

- LLMs help patchwork thinkers more than highly ordered thinkers.

  - The former jump between things with connections, and LLMs help do that for quickly.

  - The latter might find that threatening, because what the LLM produces is more work to do to verify because it's outside their ordered approach.

  - As models become able to go beyond what a grad student could do, they get harder for a non-expert on a given topic to evaluate its quality.

- The same origin paradigm: the origin can do whatever it wants with data it has access to.

  - Notably this includes flinging whatever data it can see to anywhere on the network.

  - So as a user you have to be very careful about what data you put in each origin.

  - This has a chilling effect up the stack of what data each entity is willing to share in the first place.

  - That leads to data hoarding, and the entity that hordes the most data the best gets all of the user activity.

- Most of the cold start problems of today's software are about the friction of acquiring the data to operate on.

- Imagine a new universe of software with different laws of physics, distributed via a normal web app.

  - On the outside, it would look like “just an app”.

    - It would have a cold start problem like any other app.

  - On the inside though it would be its own pocket universe.

    - Use cases could collide and grow and evolve with minimal friction.

    - Zero distribution costs within the ecosystem; use cases have minimal cold start problems because they can get the data they need to operate without any friction.

  - The “app” here is like a ballon around an emergent swarm of activity.

  - The app’s momentum would be wildly unlike a normal, non-ecosystem app.

- [<u>BIackboard</u> <u>systems</u>](https://en.wikipedia.org/wiki/Blackboard_system) are powerful emergent systems.

  - Swarms of relatively dumb little agents all emergently collaborating on a shared work space.

    - Emergently accreting complex answers to problems.

  - The power of the blackboard model is a diverse set of simple agents on a fully shared blackboard.

  - If you don’t have a shared blackboard that all of the agents can see then it doesn’t work.

  - If sharing the blackboard is scary then you’ll preclude most of the possibility space.

    - You’ll either get far fewer agents (you have to trust the agent, a high bar to clear)

    - Or you’ll get sharded blackboards, which allow only limited snapshots of the problem.

  - The same origin model fundamentally curtails the possibility of a blackboard AI system.

- Imagine: a tool for hobbyists, no matter their hobby.

  - Hobbyists are hyper motivated, very niche, collaborative.

  - The cost to create high-quality software to do a given hobby is too high, so today very little exists–generally only if one of the people with that hobby happens to be an engineer with the time to build something open source.

  - But if someone could figure out how to allow hobbyists to grow their own highly bespoke tools, it could be super useful!

- In new ecosystems everyone asks questions like “what’s the incentive for the new creators?”

  - This is often a good question, but not always.

  - For example, imagine a dynamic where a savvy user’s actions in the system to solve their own problems also indirectly create value for other users.

    - For example, there’s a brand new pop star, and a search engine doesn’t yet know that their name is something that people will want to see pictures of.

    - But a savvy user, on unexpectedly seeing no images for their query for that popstar’s name, can prepend “images of” to their query to return images.

    - Now the search engine could notice the elevated proportion of “images of X” queries for that X, and start showing images for just the query “X” too.

    - Everyone benefits from the small, in-context actions of a savvy user just trying to solve their own problem.

  - In these examples, the question of “what’s the incentive of the savvy user to create” is meaningless; they’re creating to solve a problem for themselves, and those creations just so happen to help others.

  - Ecosystems with this dynamic can bootstrap significantly faster than ecosystems without it.

- If you are not paying for your own compute then the compute isn't working for you.

  - The computation primarily works to advance the interest of the person paying the bill for the computation.

- When a human needs to read code regularly it needs to be beautiful.

  - But if a human never needs to read it, it just needs to work.

- If the creator doesn't have fun with it, then nobody else will either.

- A good characteristic of an engaging novel system. "Whoa, that worked?! What else can I do??"

  - As users tinker and experiment, they get boosts of energy to help them dive even deeper.

- Just knowing the secret doesn’t matter.

  - It’s being able to *execute* on the secret that matters.

  - Secrets often need not just knowledge, but *knowhow* to execute.

- If your thing looks superficially like existing alternatives, it’s hard to get going.

  - There isn’t enough of a gradient to differentiate yourself.

  - But if your thing has an internal network effect that the other things lack, then over time that network effect will get so much momentum that it obviates everything else.

  - If you have a secret that creates a new kind of network effect, it all comes down to: does that network effect work?

  - Even if other people try to talk about your secret and claim it as their own, it will all come to who can execute the secret to create the network effect.

  - Another downside of making a new thing look like existing things: if you make your thing look like another existing thing, people will assume it works the same way in other dimensions too, which can lead to confusion when it doesn't.

- An approach for growing a high-engagement ecosystem.

  - You want low engagement users to not even think about the ecosystem until there is so much vibrancy they’ll be high engagement, because there will be something happening they care to participate in.

  - A lame ecosystem is one where there are lots of people just standing around.

    - All of those people watching put a chilling effect on the people creating.

    - If you have a big splashy PR or marketing driven influx, then most of those new people will be low engagement.

    - They come, see everyone standing around awkwardly, and leave, never to return.

  - But if you can make it so the next person to join is the next most motivated person, you can have a high-engagement ecosystem that continually pulls in lower engagement people who will now be high engagement as the momentum and vibrancy picks up.

- There's enormously more clarity created from *doing* the thing than *talking* about the thing.

  - Talking about the thing is like a low-fi simulation of the thing.

  - But the details matter for whether a thing will be viable or not, and only the real world has the fractal, complex details.

  - Our brains can't comprehend all of the details, so they have to think in concepts (which must be an abstraction), but the real world can simulate all of the details instantly.

  - The real world is like a conversation partner that can fundamentally reason through every fractal detail instantly and tell you where it doesn’t work.

  - That’s why doing is more powerful than talking.

- When innovating, it’s more important to prove to yourself that there is *a* plausible path forward, not that this is *the* path*.*

  - The former is what determines if this step, if viable, will not put you into a dead end.

  - You don’t need to find the right path out; you just need to prove to yourself there at least is *one*.

  - But by explaining that one to others, they will cling to it too tightly as *the* path.

  - Now, if you need to adapt or change, you have to convince everyone else why you *aren’t* doing the previous plan.

- Plans are liabilities.

  - They are about talking not doing.

  - They make adapting harder.

  - When you make a plan for management to convince them you know what you’re doing, you’re locked into that even if it turns out to be wrong.

    - “How well are you executing according to the plan?

    - “Well we realized the plan is wrong”

    - “Prove to me the new plan is now correct”

    - A ton of work!

  - The over-planning that people in organizations are compelled to do is a dangerous and expensive security blanket to cover their asses and prevent meddling by their management chain.

- Given a thing X you know you want to do in the future, *when* should you do it?

  - That’s a hard question that requires some level of precision.

    - Precision that is likely both expensive and also misleading.

  - But an easier question is, “should I do X now or in the future?”

    - That’s a much easier question to answer.

    - Often the answer is simply “not yet”, you can punt it down the road, and stop thinking about it for now.

- Very few things in life should actually be maximized.

  - Most things should be satisficed.

  - We live in a finite world, not an infinite one.

  - If it’s an end in and of itself, it might make sense to maximize it.

  - If it’s merely a means, it only ever makes sense to satisfice.

- Email and tech is a means to an end.

  - People don't like email, they tolerate it as a sometimes useful means to an end.

  - The goal is not "dealing with email more effectively".

  - It’s “how can technology help people grow in ways that are meaningful to them?”

- If you're process focused, then there's never a good enough for the process, because the process is an end in and of itself.

  - If you're solution oriented, then you can hit a "good enough" point where you can correctly focus attention on other more important things.

- If you take two distinct coherent ideas and average them you get mush.

  - A synthesis is not an average, or a consensus. It is an outcome that is *better* than any of the inputs.

  - An average just pulls you to the mushy middle, the centroid.

  - A synthesis, in contrast, transcends.

  - Entropy averages until everything is the most uninteresting mush.

  - Interesting means something that stands out prominent from the background noise.

  - The bad form of consensus is about averaging.

  - The enlightened version of consensus is one that seeks disconfirming evidence and interesting eddies of gradient to surf to achieve a distinct, interesting, valuable North Star vision.

  - An auteur choosing which interesting details to absorb is way better than a group decision made by consensus.

  - There has to be someone applying judgment and making the call, otherwise the call can't be anything interesting or principled.

- Some things get stronger with more variance, but most things get duller and more mushy.

  - The auteur’s vision is what helps the idea get stronger and more powerful the more disconfirming evidence it absorbs.

- Feynman apparently recommended people to develop a habit of always guesstimating results before calculating the answer.

  - In ambiguity, the person with the better gut instincts, the judgment to discern the correct answer quickly and intuitively, will have a massive advantage.

    - Effectively a massively faster OODA loop.

  - Your gut instinct can be honed and made significantly better through practice.

  - Develop the muscle of predicting and then being surprised, and using the surprise to calibrate your predictive ability in that context.

  - Use reality to sharpen your gut instinct.

- “Move fast and break things” works great when there are no indirect effects.

  - For example, when your org is small and the product has few users, there are fewer indirect effects.

  - Moving fast and breaking things is dangerous when there are lots of indirect effects (e.g. externalities, or long feedback loops that are hard to steer).

- Building trust is an investment.

  - Short term cost for long term gain.

  - You have to plan to work with the people you’re building trust with again.

  - If you don't, then it's not worth investing in, because it will be all cost, no gain.

  - An org with very common regorgs undermines trust, because it makes it irrational to invest in trust.

- You can’t start a leading by gardening approach for a team in wartime.

  - But teams that have grown in a gardening style in peacetime are better set up for wartime if it comes.

  - Leading by gardening is about growing trust, and trust takes time to grow before it yields dividends.

  - High trust is what makes high performing teams that can succeed in extremely challenging conditions.

- An insight from a friend: "One weird trick to unlock human productivity multipliers: optimize for cooperative play"

  - If meaning-making is fun and low stakes, then the group will dive into it and learn together.

  - Normally getting diverse teams to work as one is hard.

  - But if you're leaning into cooperative play, you build trust and explore ideas together quickly.

- To have a high performing team, one way is to have everyone speak with the same voice.

  - But this gives you mush, group think.

    - Superficially safe results but in a way that is deeply dangerous.

  - Another approach is everyone having a strong individual voice that harmonizes into a transcendent whole that is way beyond what any individual could have done themselves.

- I found this [<u>interesting take on making a collaborative writers room</u>](https://okbjgm.weebly.com/uploads/3/1/5/0/31506003/11_laws_of_showrunning_nice_version.pdf) via Simon Willison.

  - Writers rooms are a better model for a collaborative, “yes, and” environment than improv, I think.

  - The writers room is about creating a particular thing, explicitly, together.

  - “Yes, and”, but with a convergent outcome, where you’re building the strongest idea together by leaning into divergent energy to find the best.

- A tell that a team has low trust: team members say "they" about other members of the team that they disagree with.

  - When someone talks about the team as "we," that means they trust the team members well enough to subjugate their ego and join with the collective.

  - That requires high trust.

- It's easier to do a transcendent synthesis of something contained inside yourself.

  - That's why absorbing disconfirming evidence, steelmanning others' positions, leaning into your contradictions is the way to finding game-changing ideas.

  - Take them in yourself and allow the synthesis to blossom.

  - Instead of seeing a team member sharing disconfirming evidence as a “they” that you want to figure out how to deal with, see them as a “we” that needs to absorb the idea to synthesize the right thing out of it.

- I love [<u>nerd clubs</u>](#s0cfteif5ebc) so much that I terraform the world around me so I can be in them as often as possible.

  - My own personal flow state requires me to be surrounded by smart, curious, open-minded people.

- In normal relationships, when you go above and beyond, the extra is banked as earned trust.

  - Which sets you up for better navigating ambiguity with them in the future, to earning a longer leash from them.

  - But an organization that demands your all, and then doesn't bank any trust is transactional, narcissistic.

    - The value flows one way, not the other way.

  - True, authentic relationships lead to mutual benefit, looking out for one another, not just transactional.

  - "Transactional" means "the interaction only looks at this interaction, not others in the past, or others in the future."

  - Transactional is greedy, it is short-sighted, finite.

- Integrity means doing the right thing even when it’s personally costly.

  - Acting with integrity earns trust.

- When you get a bit of feedback from a manager, "cut off your arm to fit in this part of the machine better".

  - Is that feedback to help you improve as a person, or to be more useful to the machine?

  - Often a given thing is good for the employee and good for the company.

  - But when they differ, is the manager operating in the interest of the machine or the employee.

    - Is the manager acting as a compassionate human with integrity, or as a meat puppet for the machine?

  - Managers in an org who show they are a person, not just an appendage of the machine, earn trust.

    - For example by giving employees advice that helps them thrive, not necessarily what is best for the machine.

- Flow state is when you're learning and growing on a dimension you *want* to grow on.

  - If the machine is telling you to grow in a dimension that is good for the machine but you suspect might be bad for you, you can't be in flow state.

- How many seeds should you plant?

  - As many as you can in the space not spent tending to your basal metabolic rate (e.g. 70% of your time to get everyone to agree your team is useful).

  - This is one of the reasons seeds need to be very cheap.

    - If they require someone to apply non-discretionary effort in their normal work time, that doesn't count as a seed, because it has a cost.

    - A seed is "a free thing this person is interested in doing over and beyond their day job."

    - The question is not “should I have them do something different” (it’s their discretionary effort!) it’s "should I allow that or not".

    - You can't redirect that energy, so it's more a freebie that you get to take or leave.

  - Seeds that don't sprout don't distract the team.

    - You only invest energy in ones that are growing, which implies that they by construction are viable.

    - That's the magic of seed planting.

- Narratively you often want to follow the inverted pyramid: put the answer first and then explain it.

  - That way, if the reader drops off at any point they’ve gotten the most bang-for-buck information.

    - Also a reader who already agrees with the answer can just drop off early, saving time.

  - But this doesn't work well for surprising questions.

  - If it's an answer to a question the reader didn't know to ask, or a non-obvious answer to a question they should be asking but aren't, then they'll bounce right off.

  - In those cases, spending just a few sentences setting up a question they likely have, that's evocative... and then giving them the surprising answer.

- People erroneously think that to have a compelling argument that captures people's attention it has to be extreme.

  - No, it has to be well argued, engaging, and clear.

  - An argument that is forcefully argued to give shape and handle to a thing you already were primed to agree with but didn't have a name for can be very compelling, provocative even.

- Why do things seem to happen in threes?

  - Because two things are not a pattern so we don’t think about any occurrences of two.

  - Three is the first thing that feels like a pattern.

  - But if it’s not a real pattern (which it often isn’t, it’s just a coincidence) then it will stop soon.

  - So we tend to find lots of patterns of three.

- You remove ambiguity by nailing things down.

  - A thing that was swirling and hard to predict is now fixed in place, something you can take for granted as you focus on the other parts.

  - But now you’re tied down and can't move if you need to!

  - The danger of nailing things down prematurely is that you lock in constraints before you even know what you want to be doing, and now you're backed into a corner.

- How do you settle down and form a web of important relationships if you're a digital nomad?

  - The ownership mindset: it's rational to engage more deeply with your neighbors if you expect to be there indefinitely, enmeshing yourself in the fabric of the community.

  - Very different between optimizing for optionality vs laying down roots.

  - Meaning comes from cost.

  - The things you decide to lay down roots with, to tie yourself deeply to something, makes it more meaningful

- If you do a vertical approach and pick the wrong vertical, you're dead.

- Many important observations are fundamentally a multi-ply argument.

  - For example, platforms that include a large collection of diverse use cases by different teams must be done with a modular building block architecture.

    - This is fundamentally true, but hard to convince people of when they’re busy and distracted.

  - 1-ply arguments are seen as "serious" and action-oriented.

    - In a chaotic environment there's no time for multi-ply arguments ever.

  - But the set of true conclusions that can be made via 1-ply arguments is wildly smaller than the set of true conclusions that can be made with an argument of any ply.

    - Another example of the streetlight fallacy, searching not where the answer is likely to lie but where searching is easiest.

  - What’s the fix to get organizations to act on multi-ply arguments at scale?

  - Have some extremely empowered leader set the first ply by fiat so everyone takes it for granted.

  - Now a large class of previously two-ply arguments become one-ply, and possible for the frenetic, swirling of the everyday experience of the organization to tackle.

  - A famous example of this working well is Bezos’ dictate that all teams had to collaborate via an API.

- As a general rule of thumb don’t take actions that make you more of a zombie.

  - Humans exercising agency is where meaning is created.

  - If you maximize just your agency then you’d steamroll and hurt others.

  - Instead take actions that will increase human agency in general now and into the future.

- SimCity allows you to garden a city.

  - You till the soil and a building might spontaneously pop up.

  - A city and a garden are very similar, it’s just that on normal time scales and from ground level you don’t see it.

  - But a simulation lets you see it at speed and perspective where you can see it.

- Creative collaboration is fun for its own sake!

  - And if you get fun, interesting, weird people together and the result turns out to be miraculous, even better!

- The folksonomy pattern only works if there's an emergent bottom up process to create it where each step is viable and compelling.

  - You can't use it to magic an ontology into existence where no interest exists.

  - A folksonomy is fundamentally an ecosystem, and like all ecosystems, you can't snap your fingers and create one.

  - You have to grow it.

- Systems are often about small but consistent asymmetries giving rise to massive macro phenomena.

  - Predictable in the macro but not in the micro.

  - The forces look like nothing but they are everything.

  - The undercurrents of the system.

- The rules of the game matter way more than the values.

  - The rules set the valid transitions from one state to the other; an intricate cage on all the possibilities.

  - The values are easier to see, the cage on possibilities of the former are hard to see; they’re the blank space in between.

  - If you don’t look carefully they’ll look like nothing at all.

- Non-systems thinking is about direct effects.

  - Only linear outcomes are possible because it’s tied to your investment of motive force.

    - Take an action, see the result.

    - Fast. Easy. Low leverage.

  - Systems thinking is about indirect effects.

    - How to make other powerful forces around you do useful work for you by tweaking something small.

    - Slow. Hard. Extraordinary leverage.

- Look at the model as an object that could be wrong, not a distillation of some objective truth.

  - In domains where there is an omnipresent powerful model, like Black Scholes, the model is so common that it fades into the background, and you stop noticing it.

  - Instead of seeing it as a thing that could be wrong, you stop thinking of it as being a thing at all, simply the objective truth.

- Beware converting non-engineering problems into engineering problems.

  - The McNamara fallacy: focus on what is measurable, and ignore what's not.

  - The streetlight fallacy: focusing only on what is easy to measure, not what matters.

    - The actions you take under the streetlight might actively harm the parts out of the streetlight.

    - You focus on the direct effects and away from the indirect effects, and you can quickly cause more harm than good.

  - If you know that the indirect effects are hard to grapple with, you're less liable to forget about them.

  - When you have a machine, you'll focus on what the machine can do and see, even if it's only a small part of the story.

  - You'll forget the other parts, and as you become increasingly blind to them, you'll steamroll right over them.

  - Some problems are complicated; they can be decomposed into an engineering problem and solved.

  - But many of the most important problems are fundamentally complex.

  - To navigate them requires grappling with their inherent complexity, dancing with it, not pretending it doesn’t exist.

- Risk and uncertainty are different

  - Risk without uncertainty is like roulette odds.

  - But most of life is dominated by uncertainty.

  - A large category error is applying risk style approaches to uncertainty.

    - Implicitly assuming a finite, well defined game.

  - Dangerously lulling yourself into a comforting illusion of certainty.

- The most important quality of a strategy is that incremental investment will lead to at least proportional incremental return.

  - If you have that shape, and there's some bonus return as you get further, that's a curve that you'll be eager to climb up at every step.

  - A smooth, non miraculous shape where each step will be a no-brainer: continuously viable as an obvious idea.

  - A swarm could hill-climb it.

  - Those kinds of ideas are way, *way* more likely to turn into real things in the world than things that have a high upfront investment for lower return.

    - They don’t require anyone to stick their neck out to jump past the adjacent possible; the emergent processes of the swarm naturally explore them.

  - Leaps past the adjacent possible are definitely possible, and if you have good taste and prediction of which ones will work then you can be very successful.

  - But there's a huge number of big investments that would more than return the investment but no one is willing to try.

- Swarms can learn even if none of the individuals do.

  - Ego gets in the way of learning hard truths, but hard truths calibrate you to innovate as an individual.

  - Hard truths are disconfirming evidence.

    - They calibrate your gut instinct.

  - And yet the swarm that finds amazing things is full of individuals with massive egos that don't learn. Why?

  - The former is about *individual* innovation.

  - The latter is about *swarm* innovation.

  - At the end of both, there will be individuals who have done amazing things.

  - The latter will be someone with a big ego that didn't learn, but that's OK because they were post-hoc selected by the swarm based on randomly winning the lottery before the start.

  - The expected value of the swarm members' tickets at the start are tiny, since most will be worth nothing.

  - As an individual, you have only one ticket, so to maximize the expected value of that ticket you need to have low ego and be willing to absorb your failures and learn.

  - As a swarm, you don't have to care about the individuals being willing to learn, as long as there's no shortage of people willing to try.

- The other day I was wondering to myself, “how did farmers figure out that Gilroy was the ideal place to grow garlic?”

  - They didn't!

  - Presumably over the last hundred or so years lots of farmers randomly tried growing garlic in lots of random places.

  - The likelihood they tried it is inversely tied to how much people expect it to fail in that location based on folk knowledge of previous failures.

  - If it didn't do well in that environment, after the season they won't do it again (and maybe will share that knowledge that it failed, directly or indirectly e.g. "Everyone knows that Steve tried garlic five years ago and it bankrupted him and he left the town in shame.").

  - After each season a farmer will decide whether to keep the same crop or swap to a different one.

  - If it went well, they're unlikely to switch.

  - But if there are other ones they think might do better they might switch.

  - Over time, the garlic farm 'grows' in the place it's best for; the other places swap to a different crop quickly.

  - A distributed percolating decision making process with no top-down component, that gives a high-quality answer that looks like it was made intelligently by a single actor but was actually decided by the swarms’ collective intelligence.

- Efficiency is vertical, at the level of the individual.

  - Resilience is horizontal, at the level of the system.

  - Resilience comes from competition of ideas and outcomes.

  - Resilience and efficiency are in tension.

- r-selection learns at the swarm, k-selected learns in the individual.

  - The logic of the swarm vs the logic of the individual.

  - R-selected organisms are much more likely than k-selected organisms to have a metamorphosis where they change from their infant to adult form.

  - Partly this is so the [<u>adults don’t compete with the kids for food</u>](https://youtu.be/ErRQN_kRQrc?si=9A4g-pxzNVbwQbLl)!

- Grasses are quite recent, evolutionarily speaking.

  - Their innovation is they [<u>grow from the bottom not from the tip</u>](https://youtu.be/B9B7kPJ1i0M?si=PTPbJpaQkJVIZZ_H), allowing them to survive being grazed or mowed.

- Plains have empires. Mountains have clans.

  - If it's a flat, frictionless plane, you get less diversity.

  - More wrinkled landscapes lead to more niches and speciation.

  - Less gene flow between adjacencies leads to more species, more diversity.

  - Literally the topology of the landscape.

  - Often figurative, but sometimes literal!

- Flowering plants dominated when they showed up on the evolutionary scene, partially because it’s faster to speciate and get new diversity.

  - For plants that pollinate via dispersal of pollen in the wind, when any pollen lands on any plant it pulls it back to the average.

  - The unrestricted gene flow prevents reproductive isolation necessary for speciation.

  - Versus flowering plants can coevolve with specific species of pollinators allowing them to be physically adjacent but still isolated, allowing faster speciation.

  - Faster speciation in niches leads to faster OODA loops.

  - Faster speciation allows not a broad resiliently-good-enough species but a lot of ones that hyper specialize to niches, and give more genetic diversity in the ecosystem.

  - An individual type of flowering plant will be less common, but the *class* of flowering plants will be much more common than any individual non-flowering plant.

  - The class has diversity that the species can't.

  - One measure of health in an ecosystem is the amount of genetic diversity.

  - The more diversity, the more likely the overall ecosystem is to have an “answer” to a new problem that pops up.

- It's hard to directly and concretely account for the indirect effects of actions.

  - But a good proxy: are you *proud* of the action you took?

  - In 10 years, if you showed it to an audience of people whose opinion you care about, would you be proud of it?

    - Merely not embarrassed?

    - Would you cringe?

  - We have an intuitive calibration of what has good indirect effects.

  - It just takes the time to take a step back and listen to it.